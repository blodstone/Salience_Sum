import datetime
import logging
import math
import os
import re
import time
import traceback
from typing import Dict, Optional, List, Tuple, Union, Iterable, Any, NamedTuple

import torch
import torch.optim.lr_scheduler
from apex import amp
from overrides import overrides

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError, parse_cuda_device
from allennlp.common.tqdm import Tqdm
from allennlp.common.util import (dump_metrics, gpu_memory_mb, peak_memory_mb,
                                  get_frozen_and_tunable_parameter_names, lazy_groups_of)
from allennlp.data.instance import Instance
from allennlp.data.iterators.data_iterator import DataIterator, TensorDict
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.nn import util as nn_util
from allennlp.training import util as training_util
from allennlp.training.checkpointer import Checkpointer
from allennlp.training.learning_rate_schedulers import LearningRateScheduler
from allennlp.training.metric_tracker import MetricTracker
from allennlp.training.momentum_schedulers import MomentumScheduler
from allennlp.training.moving_average import MovingAverage
from allennlp.training.optimizers import Optimizer
from allennlp.training.tensorboard_writer import TensorboardWriter
from allennlp.training.trainer_base import TrainerBase
from allennlp.training.trainer import Trainer
from allennlp.training.trainer_pieces import TrainerPieces

logger = logging.getLogger(__name__)


@TrainerBase.register("fp16-trainer")
class FP16Trainer(Trainer):
    def __init__(self,
                 model: Model,
                 optimizer: torch.optim.Optimizer,
                 iterator: DataIterator,
                 train_dataset: Iterable[Instance],
                 validation_dataset: Optional[Iterable[Instance]] = None,
                 patience: Optional[int] = None,
                 validation_metric: str = "-loss",
                 validation_iterator: DataIterator = None,
                 shuffle: bool = True,
                 num_epochs: int = 20,
                 serialization_dir: Optional[str] = None,
                 num_serialized_models_to_keep: int = 20,
                 keep_serialized_model_every_num_seconds: int = None,
                 checkpointer: Checkpointer = None,
                 model_save_interval: float = None,
                 cuda_device: Union[int, List] = -1,
                 grad_norm: Optional[float] = None,
                 grad_clipping: Optional[float] = None,
                 learning_rate_scheduler: Optional[LearningRateScheduler] = None,
                 momentum_scheduler: Optional[MomentumScheduler] = None,
                 summary_interval: int = 100,
                 histogram_interval: int = None,
                 should_log_parameter_statistics: bool = True,
                 should_log_learning_rate: bool = False,
                 log_batch_size_period: Optional[int] = None,
                 moving_average: Optional[MovingAverage] = None,
                 mixed_precision: bool = False) -> None:
        super().__init__(
            model,
            optimizer,
            iterator,
            train_dataset,
            validation_dataset,
            patience,
            validation_metric,
            validation_iterator,
            shuffle,
            num_epochs,
            serialization_dir,
            num_serialized_models_to_keep,
            keep_serialized_model_every_num_seconds,
            checkpointer,
            model_save_interval,
            cuda_device,
            grad_norm,
            grad_clipping,
            learning_rate_scheduler,
            momentum_scheduler,
            summary_interval,
            histogram_interval,
            should_log_parameter_statistics,
            should_log_learning_rate,
            log_batch_size_period,
            moving_average
        )

        self._mixed_precision = mixed_precision

        if self._mixed_precision:
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level="O2")

    @overrides
    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Trains one epoch and returns metrics.
        """
        logger.info("Epoch %d/%d", epoch, self._num_epochs - 1)
        peak_cpu_usage = peak_memory_mb()
        logger.info(f"Peak CPU memory usage MB: {peak_cpu_usage}")
        gpu_usage = []
        for gpu, memory in gpu_memory_mb().items():
            gpu_usage.append((gpu, memory))
            logger.info(f"GPU {gpu} memory usage MB: {memory}")

        train_loss = 0.0
        # Set the model to "train" mode.
        self.model.train()

        num_gpus = len(self._cuda_devices)

        # Get tqdm for the training batches
        raw_train_generator = self.iterator(self.train_data,
                                            num_epochs=1,
                                            shuffle=self.shuffle)
        train_generator = lazy_groups_of(raw_train_generator, num_gpus)
        num_training_batches = math.ceil(self.iterator.get_num_batches(self.train_data)/num_gpus)
        self._last_log = time.time()
        last_save_time = time.time()

        batches_this_epoch = 0
        if self._batch_num_total is None:
            self._batch_num_total = 0

        histogram_parameters = set(self.model.get_parameters_for_histogram_tensorboard_logging())


        logger.info("Training")
        train_generator_tqdm = Tqdm.tqdm(train_generator,
                                         total=num_training_batches)
        cumulative_batch_size = 0
        for batch_group in train_generator_tqdm:
            batches_this_epoch += 1
            self._batch_num_total += 1
            batch_num_total = self._batch_num_total

            self.optimizer.zero_grad()

            loss = self.batch_loss(batch_group, for_training=True)

            if torch.isnan(loss):
                raise ValueError("nan loss encountered")

            if self._mixed_precision:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            train_loss += loss.item()

            batch_grad_norm = self.rescale_gradients()

            # This does nothing if batch_num_total is None or you are using a
            # scheduler which doesn't update per batch.
            if self._learning_rate_scheduler:
                self._learning_rate_scheduler.step_batch(batch_num_total)
            if self._momentum_scheduler:
                self._momentum_scheduler.step_batch(batch_num_total)

            if self._tensorboard.should_log_histograms_this_batch():
                # get the magnitude of parameter updates for logging
                # We need a copy of current parameters to compute magnitude of updates,
                # and copy them to CPU so large models won't go OOM on the GPU.
                param_updates = {name: param.detach().cpu().clone()
                                 for name, param in self.model.named_parameters()}
                self.optimizer.step()
                for name, param in self.model.named_parameters():
                    param_updates[name].sub_(param.detach().cpu())
                    update_norm = torch.norm(param_updates[name].view(-1, ))
                    param_norm = torch.norm(param.view(-1, )).cpu()
                    self._tensorboard.add_train_scalar("gradient_update/" + name,
                                                       update_norm / (param_norm + 1e-7))
            else:
                self.optimizer.step()

            # Update moving averages
            if self._moving_average is not None:
                self._moving_average.apply(batch_num_total)

            # Update the description with the latest metrics
            metrics = training_util.get_metrics(self.model, train_loss, batches_this_epoch)
            description = training_util.description_from_metrics(metrics)

            train_generator_tqdm.set_description(description, refresh=False)

            # Log parameter values to Tensorboard
            if self._tensorboard.should_log_this_batch():
                self._tensorboard.log_parameter_and_gradient_statistics(self.model, batch_grad_norm)
                self._tensorboard.log_learning_rates(self.model, self.optimizer)

                self._tensorboard.add_train_scalar("loss/loss_train", metrics["loss"])
                self._tensorboard.log_metrics({"epoch_metrics/" + k: v for k, v in metrics.items()})

            if self._tensorboard.should_log_histograms_this_batch():
                self._tensorboard.log_histograms(self.model, histogram_parameters)

            if self._log_batch_size_period:
                cur_batch = sum([training_util.get_batch_size(batch) for batch in batch_group])
                cumulative_batch_size += cur_batch
                if (batches_this_epoch - 1) % self._log_batch_size_period == 0:
                    average = cumulative_batch_size/batches_this_epoch
                    logger.info(f"current batch size: {cur_batch} mean batch size: {average}")
                    self._tensorboard.add_train_scalar("current_batch_size", cur_batch)
                    self._tensorboard.add_train_scalar("mean_batch_size", average)

            # Save model if needed.
            if self._model_save_interval is not None and (
                    time.time() - last_save_time > self._model_save_interval
            ):
                last_save_time = time.time()
                self._save_checkpoint(
                        '{0}.{1}'.format(epoch, training_util.time_to_str(int(last_save_time)))
                )
        metrics = training_util.get_metrics(self.model, train_loss, batches_this_epoch, reset=True)
        metrics['cpu_memory_MB'] = peak_cpu_usage
        for (gpu_num, memory) in gpu_usage:
            metrics['gpu_'+str(gpu_num)+'_memory_MB'] = memory
        torch.cuda.empty_cache()
        return metrics

    @classmethod
    def from_params(cls,  # type: ignore
                    params: Params,
                    serialization_dir: str,
                    recover: bool = False,
                    cache_directory: str = None,
                    cache_prefix: str = None):
        pieces = TrainerPieces.from_params(params, serialization_dir, recover)
        model = pieces.model
        iterator = pieces.iterator
        train_data = pieces.train_dataset
        validation_data = pieces.validation_dataset
        params = pieces.params
        validation_iterator = pieces.validation_iterator

        patience = params.pop_int("patience", None)
        validation_metric = params.pop("validation_metric", "-loss")
        shuffle = params.pop_bool("shuffle", True)
        num_epochs = params.pop_int("num_epochs", 20)
        cuda_device = parse_cuda_device(params.pop("cuda_device", -1))
        grad_norm = params.pop_float("grad_norm", None)
        grad_clipping = params.pop_float("grad_clipping", None)
        lr_scheduler_params = params.pop("learning_rate_scheduler", None)
        momentum_scheduler_params = params.pop("momentum_scheduler", None)

        if isinstance(cuda_device, list):
            model_device = cuda_device[0]
        else:
            model_device = cuda_device
        if model_device >= 0:
            # Moving model to GPU here so that the optimizer state gets constructed on
            # the right device.
            model = model.cuda(model_device)

        parameters = [[n, p] for n, p in model.named_parameters() if p.requires_grad]
        optimizer = Optimizer.from_params(parameters, params.pop("optimizer"))
        if "moving_average" in params:
            moving_average = MovingAverage.from_params(params.pop("moving_average"), parameters=parameters)
        else:
            moving_average = None

        if lr_scheduler_params:
            lr_scheduler = LearningRateScheduler.from_params(optimizer, lr_scheduler_params)
        else:
            lr_scheduler = None
        if momentum_scheduler_params:
            momentum_scheduler = MomentumScheduler.from_params(optimizer, momentum_scheduler_params)
        else:
            momentum_scheduler = None

        if 'checkpointer' in params:
            if 'keep_serialized_model_every_num_seconds' in params or \
                    'num_serialized_models_to_keep' in params:
                raise ConfigurationError(
                    "Checkpointer may be initialized either from the 'checkpointer' key or from the "
                    "keys 'num_serialized_models_to_keep' and 'keep_serialized_model_every_num_seconds'"
                    " but the passed config uses both methods.")
            checkpointer = Checkpointer.from_params(params.pop("checkpointer"))
        else:
            num_serialized_models_to_keep = params.pop_int("num_serialized_models_to_keep", 20)
            keep_serialized_model_every_num_seconds = params.pop_int(
                "keep_serialized_model_every_num_seconds", None)
            checkpointer = Checkpointer(
                serialization_dir=serialization_dir,
                num_serialized_models_to_keep=num_serialized_models_to_keep,
                keep_serialized_model_every_num_seconds=keep_serialized_model_every_num_seconds)
        model_save_interval = params.pop_float("model_save_interval", None)
        summary_interval = params.pop_int("summary_interval", 100)
        histogram_interval = params.pop_int("histogram_interval", None)
        should_log_parameter_statistics = params.pop_bool("should_log_parameter_statistics", True)
        should_log_learning_rate = params.pop_bool("should_log_learning_rate", False)
        log_batch_size_period = params.pop_int("log_batch_size_period", None)
        mixed_precision = params.pop_bool("mixed_precision", False)

        params.assert_empty(cls.__name__)
        return cls(model, optimizer, iterator,
                   train_data, validation_data,
                   patience=patience,
                   validation_metric=validation_metric,
                   validation_iterator=validation_iterator,
                   shuffle=shuffle,
                   num_epochs=num_epochs,
                   serialization_dir=serialization_dir,
                   cuda_device=cuda_device,
                   grad_norm=grad_norm,
                   grad_clipping=grad_clipping,
                   learning_rate_scheduler=lr_scheduler,
                   momentum_scheduler=momentum_scheduler,
                   checkpointer=checkpointer,
                   model_save_interval=model_save_interval,
                   summary_interval=summary_interval,
                   histogram_interval=histogram_interval,
                   should_log_parameter_statistics=should_log_parameter_statistics,
                   should_log_learning_rate=should_log_learning_rate,
                   log_batch_size_period=log_batch_size_period,
                   moving_average=moving_average,
                   mixed_precision=mixed_precision)
