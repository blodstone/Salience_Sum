import logging
import os

import torch

from allennlp.data import Vocabulary
from allennlp.data.iterators import BucketIterator
from allennlp.data.tokenizers import WordTokenizer
from allennlp.data.tokenizers.word_splitter import JustSpacesWordSplitter
from allennlp.modules import Embedding
from allennlp.modules.seq2seq_decoders import LstmCellDecoderNet, AutoRegressiveSeqDecoder
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.training import Trainer
from torch import optim

from salience_sum.model import DenoisingEncoder
from salience_sum.model.salience_model import SalienceSeq2Seq
from salience_sum.reader import SummDataReader

EMBEDDING_DIM = 128
HIDDEN_DIM = 64

class Test:

    def setUp(self):
        logging.basicConfig(level=logging.INFO)

        tokenizer = WordTokenizer(JustSpacesWordSplitter())
        reader = SummDataReader(tokenizer, source_max_tokens=400, lazy=False)
        self.train_dataset = reader.read('../data/dev_bbc/train.dev.tsv.tagged')
        self.val_dataset = reader.read('../data/dev_bbc/val.dev.tsv.tagged')
        vocab_path = 'data/cnndm/vocab'
        if os.path.exists(vocab_path):
            self.vocab = Vocabulary.from_files(vocab_path)
        else:
            self.vocab = Vocabulary.from_instances(self.train_dataset, max_vocab_size=80000)
            self.vocab.save_to_files(vocab_path)

    def test_reader(self):
        tokenizer = WordTokenizer(JustSpacesWordSplitter())
        reader = SummDataReader(tokenizer=tokenizer, source_max_tokens=400, target_max_tokens=100)
        train_dataset = reader.read('../data/dev_bbc/train.dev.tsv.tagged')
        vocab = Vocabulary.from_instances(train_dataset)
        assert vocab.get_vocab_size('tokens') > 2

    def test_model(self):
        self.setUp()
        embedding = Embedding(
            num_embeddings=self.vocab.get_vocab_size('tokens'),
            embedding_dim=EMBEDDING_DIM)
        embedder = BasicTextFieldEmbedder({'tokens': embedding})
        encoder = PytorchSeq2SeqWrapper(
            DenoisingEncoder(bidirectional=True, num_layers=2,
                             input_size=EMBEDDING_DIM, hidden_size=HIDDEN_DIM,
                             use_bridge=True))
        decoder_net = LstmCellDecoderNet(decoding_dim=HIDDEN_DIM, target_embedding_dim=EMBEDDING_DIM)
        decoder = AutoRegressiveSeqDecoder(
            max_decoding_steps=100, target_namespace='tokens',
            target_embedder=embedding, beam_size=5, decoder_net=decoder_net, vocab=self.vocab)
        model = SalienceSeq2Seq(encoder=encoder, decoder=decoder, vocab=self.vocab, source_text_embedder=embedder)
        optimizer = optim.Adam(model.parameters(), lr=0.1)
        iterator = BucketIterator(batch_size=4, sorting_keys=[("source_tokens", "num_tokens")])
        iterator.index_with(self.vocab)
        if torch.cuda.is_available():
            cuda_device = 0
            model = model.cuda(cuda_device)
        else:
            cuda_device = -1
        trainer = Trainer(model=model, optimizer=optimizer, train_dataset=self.train_dataset,
                          validation_dataset=self.val_dataset, iterator=iterator, num_epochs=2, cuda_device=cuda_device)
        trainer.train()
