from typing import Dict

import torch
import torch.nn as nn
import torch.distributions as D
import torch.nn.functional as F
from allennlp.common import Registrable
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.nn import RegularizerApplicator


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
                 bidirectional_input=False) -> None:
        super().__init__()
        self._bidirectional_input = bidirectional_input
        self.criterion = nn.MSELoss()
        self.regression = torch.nn.Linear(hidden_dim, 1, bias=True)

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return super().get_metrics(reset)

    def KL_div(self, a, b):
        log_a = a.log()
        log_b = b.log()
        log_a[log_a == float('-inf')] = 0
        log_b[log_b == float('-inf')] = 0
        return (b * (log_b - log_a)).sum()

    def _get_loss(self, predicted_salience, salience_values) -> Dict[str, torch.Tensor]:
        # Jensen Shannon Divergence
        m = (predicted_salience + salience_values) * 0.5
        distance = 0.5 * self.KL_div(predicted_salience, m) + 0.5 * self.KL_div(salience_values, m)
        return distance

    def forward(self,
                encoder_out: Dict[str, torch.LongTensor],
                salience_values
                ) -> Dict[str, torch.Tensor]:

        # shape: (batch_size, seq_len, 1)
        regression_output = self.regression(encoder_out['encoder_outputs'])
        predicted_salience = torch.relu(regression_output).squeeze(dim=2) + 1e-6
        loss = torch.sqrt(self.criterion(predicted_salience, salience_values))
        if torch.isnan(loss):
            raise ValueError("nan loss encountered")
        output_dict = {
            'loss': loss
        }
        return output_dict
        # return self._forward_loss(predicted_salience, salience_values)
