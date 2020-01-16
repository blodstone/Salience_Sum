from typing import Dict, Optional, Tuple

import torch
from allennlp.modules.seq2seq_decoders import DecoderNet
import torch.nn as nn


class TransformerDecoder(DecoderNet):


    def __init__(self,
                 decoding_dim: int,
                 target_embedding_dim: int,
                 feedforward_hidden_dim: int,
                 num_layers: int,
                 num_attention_heads: int,
                 dropout_prob: float = 0.1,
                 ) -> None:
        super().__init__(decoding_dim=decoding_dim,
                         target_embedding_dim=target_embedding_dim,
                         decodes_parallel=True)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=decoding_dim,
            nhead=num_attention_heads,
            dim_feedforward=feedforward_hidden_dim,
            dropout=dropout_prob
        )
        decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

    def forward(self, previous_state: Dict[str, torch.Tensor], encoder_outputs: torch.Tensor, source_mask: torch.Tensor,
                previous_steps_predictions: torch.Tensor, previous_steps_mask: Optional[torch.Tensor] = None) -> Tuple[
        Dict[str, torch.Tensor],
        torch.Tensor]:
        pass

    def init_decoder_state(self, encoder_out: Dict[str, torch.LongTensor]) -> Dict[str, torch.Tensor]:
        pass

