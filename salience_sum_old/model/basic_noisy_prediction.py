from typing import Dict

import torch
import torch.nn as nn
import torch.distributions as D
import torch.nn.functional as F
from allennlp.common import Registrable
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.nn import RegularizerApplicator
from allennlp.training.metrics import CategoricalAccuracy


class NoisyPredictionModel(nn.Module, Registrable):
    default_implementation = 'basic_noisy_prediction'

    def __init__(self) -> None:
        super().__init__()

    def get_output_dim(self) -> int:
        """
        The dimension of each timestep of the hidden state in the layer before final softmax.
        Needed to check whether the model is compaitble for embedding-final layer weight tying.
        """
        raise NotImplementedError()

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        """
        The decoder is responsible for computing metrics using the target tokens.
        """
        raise NotImplementedError()

    def forward(self,
                encoder_out: Dict[str, torch.LongTensor],
                salience_values: torch.FloatTensor) -> Dict[str, torch.Tensor]:
        """
        Decoding from encoded states to sequence of outputs
        also computes loss if ``target_tokens`` are given.

        Parameters
        ----------
        encoder_out : ``Dict[str, torch.LongTensor]``, required
            Dictionary with encoded state, ideally containing the encoded vectors and the
            source mask.
        target_tokens : ``Dict[str, torch.LongTensor]``, optional
            The output of `TextField.as_array()` applied on the target `TextField`.

       """
        # pylint: disable=arguments-differ
        raise NotImplementedError()

    def post_process(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
            Post processing for converting raw outputs to prediction during inference.
            The composing models such ``allennlp.models.encoder_decoders.composed_seq2seq.ComposedSeq2Seq``
            can call this method when `decode` is called.
        """
        raise NotImplementedError()


@NoisyPredictionModel.register('basic_noisy_prediction')
class BasicNoisyPredictionModel(nn.Module, Registrable):

    def __init__(self,
                 hidden_dim: int,
                 proj_dim: int,
                 bidirectional_input=False) -> None:
        super().__init__()
        self._bidirectional_input = bidirectional_input
        self.criterion = nn.BCELoss()
        self.projection = torch.nn.Linear(hidden_dim, proj_dim, bias=True)
        self.activation_1 = torch.nn.ReLU()
        self.regression = torch.nn.Linear(proj_dim, 1)
        self.activation_2 = torch.nn.Sigmoid()
        # with torch.no_grad():
        #     self.regression.weight.data = torch.nn.init.kaiming_normal_(
        #         torch.empty(1, hidden_dim), mode='fan_out', nonlinearity='leaky_relu')
        self.loss = torch.tensor([0])

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"BCE": self.loss.item()}

    def forward(self,
                encoder_out: Dict[str, torch.LongTensor],
                salience_values
                ) -> Dict[str, torch.Tensor]:

        # shape: (batch_size, seq_len, 1)
        projection = self.activation_1(self.projection(encoder_out['encoder_outputs']))
        regression_output = self.regression(projection)
        predicted_salience = self.activation_2(regression_output.squeeze(dim=2))
        # predicted_salience = regression_output.squeeze(dim=2)
        self.loss = self.criterion(predicted_salience, salience_values)
        if torch.isnan(self.loss):
            raise ValueError("nan loss encountered")
        output_dict = {
            'loss': self.loss,
            'pred_salience': predicted_salience
        }
        return output_dict
        # return self._forward_loss(predicted_salience, salience_values)
