import logging
import os

import torch
from allennlp.data import Vocabulary
from allennlp.data.iterators import BucketIterator
from allennlp.data.tokenizers import WordTokenizer
from allennlp.data.tokenizers.word_splitter import JustSpacesWordSplitter
from allennlp.models import ComposedSeq2Seq
from allennlp.modules import Embedding
from allennlp.modules.seq2seq_decoders import LstmCellDecoderNet, AutoRegressiveSeqDecoder
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.training import Trainer
from torch import optim

from model.denoising_encoder import DenoisingEncoder
from model.noisy_prediction import NoisyPredictionModel
from model.salience_model import SalienceSeq2Seq
from reader.summ_data_reader import SummDataReader

if __name__ == '__main__':
    """
    Initial Setup and Variable Definition
    """
    logging.basicConfig(level=logging.INFO)
    EMBEDDING_DIM = 128
    HIDDEN_DIM = 64

    """
    Setup Reader and Vocab
    """
    tokenizer = WordTokenizer(JustSpacesWordSplitter())
    reader = SummDataReader(tokenizer, source_max_tokens=400, lazy=False)
    train_dataset = reader.read('data/dev_bbc/train.dev.tsv.tagged')
    val_dataset = reader.read('data/dev_bbc/val.dev.tsv.tagged')
    vocab_path = 'data/dev_bbc/vocab'
    if os.path.exists(vocab_path):
        vocab = Vocabulary.from_files(vocab_path)
    else:
        vocab = Vocabulary.from_instances(train_dataset, max_vocab_size=80000)
        vocab.save_to_files(vocab_path)

    """
    Setup Model
    """
    embedding = Embedding(
        num_embeddings=vocab.get_vocab_size('tokens'),
        embedding_dim=EMBEDDING_DIM)
    embedder = BasicTextFieldEmbedder({'tokens': embedding})
    encoder = PytorchSeq2SeqWrapper(
        DenoisingEncoder(bidirectional=True, num_layers=2,
                         input_size=EMBEDDING_DIM, hidden_size=HIDDEN_DIM,
                         use_bridge=True))
    decoder_net = LstmCellDecoderNet(decoding_dim=HIDDEN_DIM, target_embedding_dim=EMBEDDING_DIM)
    decoder = AutoRegressiveSeqDecoder(
        max_decoding_steps=100, target_namespace='tokens',
        target_embedder=embedding, beam_size=5, decoder_net=decoder_net, vocab=vocab)
    noisy_prediction = NoisyPredictionModel(vocab=vocab, hidden_dim=HIDDEN_DIM)
    model = SalienceSeq2Seq(noisy_prediction=noisy_prediction, encoder=encoder, decoder=decoder, vocab=vocab, source_text_embedder=embedder)

    """
    Setup Trainer
    """
    optimizer = optim.Adam(model.parameters(), lr=0.1)
    iterator = BucketIterator(batch_size=4, sorting_keys=[("source_tokens", "num_tokens")])
    iterator.index_with(vocab)
    if torch.cuda.is_available():
        cuda_device = 0
        model = model.cuda(cuda_device)
    else:
        cuda_device = -1
    trainer = Trainer(model=model, optimizer=optimizer, train_dataset=train_dataset,
                      validation_dataset=val_dataset, iterator=iterator, num_epochs=2, cuda_device=cuda_device)
    trainer.train()
