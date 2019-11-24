from typing import Dict, Optional
import torch
from allennlp.nn.util import get_text_field_mask
from overrides import overrides

from allennlp.common.checks import ConfigurationError
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder, Embedding, InputVariationalDropout
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.seq2seq_decoders.seq_decoder import SeqDecoder
from allennlp.nn import util, InitializerApplicator, RegularizerApplicator

from salience_sum.model import NoisyPredictionModel


@Model.register("salience_seq2seq")
class SalienceSeq2Seq(Model):
    """
    This ``ComposedSeq2Seq`` class is a :class:`Model` which takes a sequence, encodes it, and then
    uses the encoded representations to decode another sequence.  You can use this as the basis for
    a neural machine translation system, an abstractive summarization system, or any other common
    seq2seq problem.  The model here is simple, but should be a decent starting place for
    implementing recent models for these tasks.

    The ``ComposedSeq2Seq`` class composes separate ``Seq2SeqEncoder`` and ``SeqDecoder`` classes.
    These parts are customizable and are independent from each other.

    Parameters
    ----------
    vocab : ``Vocabulary``, required
        Vocabulary containing source and target vocabularies. They may be under the same namespace
        (`tokens`) or the target tokens can have a different namespace, in which case it needs to
        be specified as `target_namespace`.
    source_text_embedders : ``TextFieldEmbedder``, required
        Embedders for source side sequences
    encoder : ``Seq2SeqEncoder``, required
        The encoder of the "encoder/decoder" model
    decoder : ``SeqDecoder``, required
        The decoder of the "encoder/decoder" model
    tied_source_embedder_key : ``str``, optional (default=``None``)
        If specified, this key is used to obtain token_embedder in `source_text_embeencoder_outdder` and
        the weights are shared/tied with the decoder's target embedding weights.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """

    def __init__(self,
                 vocab: Vocabulary,
                 source_text_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 decoder: SeqDecoder,
                 noisy_prediction: NoisyPredictionModel,
                 dropout: float = 0.2,
                 tied_source_embedder_key: Optional[str] = None,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:

        super(SalienceSeq2Seq, self).__init__(vocab, regularizer)

        self._source_text_embedder = source_text_embedder
        self._encoder = encoder
        self._decoder = decoder
        self._dropout = InputVariationalDropout(dropout)
        self._noisy_prediction = noisy_prediction

        if self._encoder.get_output_dim() != self._decoder.get_output_dim():
            raise ConfigurationError(f"Encoder output dimension {self._encoder.get_output_dim()} should be"
                                     f" equal to decoder dimension {self._decoder.get_output_dim()}.")
        if tied_source_embedder_key:
            # A bit of a ugly hack to tie embeddings.
            # Works only for `BasicTextFieldEmbedder`, and since
            # it can have multiple embedders, and `SeqDecoder` contains only a single embedder, we need
            # the key to select the source embedder to replace it with the target embedder from the decoder.
            if not isinstance(self._source_text_embedder, BasicTextFieldEmbedder):
                raise ConfigurationError("Unable to tie embeddings,"
                                         "Source text embedder is not an instance of `BasicTextFieldEmbedder`.")
            # pylint: disable=protected-access
            source_embedder = self._source_text_embedder._token_embedders[tied_source_embedder_key]
            if not isinstance(source_embedder, Embedding):
                raise ConfigurationError("Unable to tie embeddings,"
                                         "Selected source embedder is not an instance of `Embedding`.")
            if source_embedder.get_output_dim() != self._decoder.target_embedder.get_output_dim():
                raise ConfigurationError(f"Output Dimensions mismatch between"
                                         f"source embedder and target embedder.")
            self._source_text_embedder._token_embedders[tied_source_embedder_key] = self._decoder.target_embedder
        initializer(self)

    @overrides
    def forward(self,  # type: ignore
                source_tokens: Dict[str, torch.LongTensor],
                target_tokens: Dict[str, torch.LongTensor] = None,
                salience_values=None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Make foward pass on the encoder and decoder for producing the entire target sequence.

        Parameters
        ----------
        source_tokens : ``Dict[str, torch.LongTensor]``
           The output of `TextField.as_array()` applied on the source `TextField`. This will be
           passed through a `TextFieldEmbedder` and then through an encoder.
        target_tokens : ``Dict[str, torch.LongTensor]``, optional (default = None)
           Output of `Textfield.as_array()` applied on target `TextField`. We assume that the
           target tokens are also represented as a `TextField`.

        Returns
        -------
        Dict[str, torch.Tensor]
            The output tensors from the decoder.
        """
        state = self._encode(source_tokens)
        salience_output = self._noisy_prediction(state, salience_values)
        decoder_output = self._decoder(state, source_tokens)
        # print("Salience loss: {}".format(decoder_output['loss']))
        # print("Decoder loss: {}".format(salience_output['loss']))
        final_output = {
            'loss': decoder_output['loss'] + salience_output['loss']
        }
        if torch.isnan(final_output['loss']):
            raise ValueError("nan loss encountered")
        # Autodecoder using source_tokens
        return final_output

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Finalize predictions.
        """
        return self._decoder.post_process(output_dict)

    def _encode(self, source_tokens: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Make foward pass on the encoder.

        Parameters
        ----------
        source_tokens : ``Dict[str, torch.LongTensor]``
           The output of `TextField.as_array()` applied on the source `TextField`. This will be
           passed through a `TextFieldEmbedder` and then through an encoder.

        Returns
        -------
        Dict[str, torch.Tensor]
            Map consisting of the key `source_mask` with the mask over the
            `source_tokens` text field,
            and the key `encoder_outputs` with the output tensor from
            forward pass on the encoder.
        """
        # shape: (batch_size, max_input_sequence_length, encoder_input_dim)
        embedded_input = self._dropout(self._source_text_embedder(source_tokens))
        # shape: (batch_size, max_input_sequence_length)
        source_mask = util.get_text_field_mask(source_tokens)
        # shape: (batch_size, max_input_sequence_length, encoder_output_dim)
        encoder_outputs = self._encoder(embedded_input, source_mask)
        return {
            "source_mask": source_mask,
            "encoder_outputs": encoder_outputs,
        }

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return self._noisy_prediction.get_metrics(reset)
