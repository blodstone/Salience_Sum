import torch
from allennlp.common import Registrable
from torch.nn import Module, Linear, Sequential, Tanh, Parameter, ModuleList
from torch.nn.functional import softmax

from pg_salience_feature.module.salience_embedder import SalienceEmbedder


class SalienceSourceMixer(Module, Registrable):
    def __init__(self,
                 embedding_size: int,
                 feature_size: int,
                 salience_embedder: SalienceEmbedder,
                 ):
        super().__init__()
        self.feature_size = feature_size
        self.embedding_size = embedding_size
        self.salience_embedder = salience_embedder
        self.linear_salience = Linear(self.hidden_size,
                                      self.hidden_size * 2,
                                      bias=False)


@SalienceSourceMixer.register('no_mix')
class NoMix(SalienceSourceMixer, Registrable):
    def forward(self, salience_values, embedded_src):
        emb_salience_value = self.salience_embedder(salience_values)
        return embedded_src, self.linear_salience(emb_salience_value)


@SalienceSourceMixer.register('emb_mlp')
class LinearConcat(SalienceSourceMixer, Registrable):
    def __init__(self, embedding_size: int,
                 feature_size: int,
                 salience_embedder: SalienceEmbedder):
        super().__init__(embedding_size, feature_size, salience_embedder)
        self.linear = Sequential(
            Linear(2 * embedding_size, 2 * embedding_size, bias=True),
            Tanh(),
            Linear(2 * embedding_size, embedding_size)
        )

    def forward(self, salience_values, embedded_src):
        emb_salience_value = self.salience_embedder(salience_values)
        embedded_src = torch.cat([embedded_src, emb_salience_value], dim=2)
        return self.linear(embedded_src), self.linear_salience(emb_salience_value)


# noinspection PyArgumentList
@SalienceSourceMixer.register('bilinear_attn')
class BilinearAttention(SalienceSourceMixer, Registrable):
    def __init__(self, embedding_size: int,
                 feature_size: int,
                 k_size: int,
                 c_size: int,
                 p_size: int,
                 glimpse: int,
                 salience_embedder: SalienceEmbedder):
        super().__init__(embedding_size, feature_size, salience_embedder)
        self.glimpse = glimpse
        self.k_size = k_size
        self.p_size = p_size
        self.c_size = c_size
        self.U = Linear(self.embedding_size, self.k_size)
        self.V = Linear(self.embedding_size, self.k_size)
        self.U_ = Linear(self.embedding_size, self.k_size)
        self.V_ = Linear(self.embedding_size, self.k_size)
        self.P = ModuleList(
            [
                Linear(1, self.k_size)
                for _ in range(self.glimpse)
            ])
        self.F = Linear(self.k_size, self.c_size)

    def bilinear_attn_map(self, i, emb_salience, embedded_src):
        batch_size = emb_salience.size(0)

        t1 = self.U_(embedded_src)
        t2 = self.V_(emb_salience)
        p = self.P[i](torch.ones(1, device=emb_salience.device))
        p = p.expand(batch_size, 1, self.k_size)
        t3 = (p[i] * t1).bmm(t2.view(-1, t2.size(2), t2.size(1)))
        A = softmax(t3.view(-1, t3.size(1) * t3.size(2)), dim=-1).view(-1, 1, self.feature_size)
        return A

    def bilinear_attn_net(self, x, y, A):
        attn_x = x.view(-1, x.size(2), x.size(1)).bmm(A)
        result = []
        for i in range(self.k_size):
            result.append(attn_x[:, 0, :].unsqueeze(1).bmm(y[:, :, 0].unsqueeze(2)))
        f_ = torch.stack(result, dim=1).squeeze(2)
        return self.F(f_.view(-1, f_.size(2), f_.size(1)))

    def forward(self, salience_values, emb_src):
        batch_size = salience_values.size(0)
        length = salience_values.size(1)
        n = batch_size * length
        emb_salience = self.salience_embedder(salience_values)
        emb_src = emb_src.unsqueeze(2)
        emb_salience = emb_salience.view(n, emb_salience.size(2), -1)
        emb_src = emb_src.view(n, emb_src.size(2), -1)
        t1 = self.U(emb_src)
        t2 = self.V(emb_salience)
        A = self.bilinear_attn_map(0, emb_salience, emb_src)
        f = self.bilinear_attn_net(t1, t2, A)
        for i in range(1, self.glimpse):
            A = self.bilinear_attn_map(i, emb_salience, emb_src)
            f = self.bilinear_attn_net(f, t2, A) + f
        return f.view(batch_size, length, -1), self.linear_salience(emb_salience)
