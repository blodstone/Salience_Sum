from allennlp.common import Registrable
import torch
from torch.nn import Module, Embedding, Linear, ReLU, Sequential, ModuleList


class SalienceEmbedder(Module, Registrable):

    def __init__(self, embedding_size: int, feature_size: int):
        super().__init__()
        self.embedding_size = embedding_size
        self.feature_size = feature_size


@SalienceEmbedder.register('matrix')
class SalienceEmbedderMatrix(SalienceEmbedder):

    def __init__(self, embedding_size: int, feature_size: int):
        super().__init__(embedding_size, feature_size)
        self.embedders = ModuleList(
            [
                Embedding(2, self.embedding_size)
                for _ in range(feature_size)
            ])

    def forward(self, salience_values: torch.Tensor):
        embs = []
        for i, salience_value in enumerate(salience_values.split(1, 2)):
            embs.append(self.embedders[i](salience_value))
        return torch.cat(embs, dim=2)


@SalienceEmbedder.register('vector')
class SalienceEmbedderVector(SalienceEmbedder):

    def __init__(self, embedding_size: int, feature_size: int):
        super().__init__(embedding_size, feature_size)
        self.embedders = Embedding(pow(2, self.feature_size), self.embedding_size)

    def forward(self, salience_values: torch.Tensor):
        embs = torch.zeros(salience_values.size(0), salience_values.size(1), 1,
                           device=salience_values.device, dtype=torch.long)
        # sum_salience_values = salience_values.sum(dim=2)
        for i, salience_value in enumerate(salience_values.split(1, 1)):
            for j, value in enumerate(salience_value.split(1, 2)):
                embs[:, i, 0] += pow(2, j) * value.squeeze()
        return self.embedders(embs).squeeze()
