from typing import Dict

import torch
import torch.nn as nn
import torch.distributions as D
import torch.nn.functional as F
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.nn import RegularizerApplicator, util


class NoisyPredictionModel(Model):

    def __init__(self,
                 vocab: Vocabulary,
                 hidden_dim: int,
                 bidirectional_input=False,
                 regularizer: RegularizerApplicator = None) -> None:
        super().__init__(vocab, regularizer)
        self._bidirectional_input = bidirectional_input
        self.criterion = nn.KLDivLoss(reduction='none')
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
        mask = encoder_out['source_mask']
        seq_len = mask.sum(1)

        # shape: (batch_size, seq_len, 1)
        regression_output = self.regression(encoder_out['encoder_outputs'])
        # Sampling dirichlet
        alphas = torch.relu(regression_output).squeeze(dim=2) + 1e-6
        d_sample = lambda x: D.Dirichlet(x).rsample()
        loss = []
        for idx, alpha in enumerate(alphas):
            predicted_salience = torch.Tensor(d_sample(alpha[:seq_len[idx]]))
            loss.append(self._get_loss(predicted_salience, salience_values[idx][:seq_len[idx]]))
        loss = torch.stack(loss).mean()
        if torch.isnan(loss):
            raise ValueError("nan loss encountered")
        output_dict = {
            'loss': loss
        }
        return output_dict
        # return self._forward_loss(predicted_salience, salience_values)
