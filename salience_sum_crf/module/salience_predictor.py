import torch
from allennlp.common import Registrable
from allennlp.common.checks import ConfigurationError
from torch.nn import Module, Sequential, Linear, ReLU, Sigmoid, Tanh, LeakyReLU
from typing import Dict, Tuple, Union


class SaliencePredictor(Module, Registrable):

    def __init__(self, hidden_size: int, bidirectional: bool):
        super().__init__()
        self.bidirectional = bidirectional
        if not self.bidirectional:
            raise ConfigurationError('Salience predictor need to be bidirectional.')
        self.hidden_size = hidden_size
        self.predict = Sequential(
            Linear(2*self.hidden_size, 2*self.hidden_size, bias=True),
            Tanh(),
            Linear(2*self.hidden_size, 2, bias=True),
            Sigmoid(),
            Linear(2, 1, bias=False),
        )

    def forward(self, state: Dict[str, Union[torch.Tensor, Tuple, Dict[str, torch.Tensor]]]):
        states = state['encoder_states']
        output = self.predict(states)
        return output
