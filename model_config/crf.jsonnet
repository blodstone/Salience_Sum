{
  "dataset_reader": {
    "type": "crfsummdatareader",
    "lazy": false,
    "interpolation": false,
    "tokenizer": {
      "type": "word",
      "word_splitter": {
        "type": "just_spaces"
      }
    },
    "source_max_tokens": 400
  },
  "train_data_path": "data/bbc_allen/train.tsv.tagged",
  "validation_data_path": "data/bbc_allen/validation.tsv.tagged",
  "model": {
    "type": "crf_tagger",
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "embedding_dim": 128
        }
      }
    },
    "encoder": {
      "type": "gru",
      "input_size": 128,
      "hidden_size": 256,
      "num_layers": 1,
      "dropout": 0.0,
      "bidirectional": true
    },
    "regularizer": [
      ["transitions$", {"type": "l2", "alpha": 0.01}]
    ]
  },
  "iterator": {"type": "basic", "batch_size": 32},
  "trainer": {
    "optimizer": "adam",
    "num_epochs": 50,
    "patience": 7,
    "cuda_device": 0
  },
  "vocabulary": {
    "max_vocab_size": 200000
  }
}
