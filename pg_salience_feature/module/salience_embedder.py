from allennlp.common import Registrable
import torch
from torch.nn import Module, Embedding, Linear, ReLU, Sequential, ModuleList


class SalienceEmbedder(Module, Registrable):

    def __init__(self, hidden_size: int, feature_size: int):
        super().__init__()
        self.embedders = ModuleList(
            [
                Sequential(
                    Linear(feature_size, hidden_size, bias=False),
                    Linear(hidden_size, hidden_size, bias=True),
                    ReLU()) for i in range(feature_size)
            ])

    def forward(self, salience_values):
        embs = []
        for i, salience_value in enumerate(salience_values.split(1, 2)):
            embs.append(self.embedders[i](salience_value))
        return torch.cat(embs, dim=2)
