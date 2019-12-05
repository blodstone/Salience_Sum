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
  "train_data_path": "/home/acp16hh/Projects/Research/Experiments/Exp_Gwen_Saliency_Summ/src/Salience_Sum/data/dev_bbc/train.dev.tsv.tagged.small",
  "validation_data_path": "/home/acp16hh/Projects/Research/Experiments/Exp_Gwen_Saliency_Summ/src/Salience_Sum/data/dev_bbc/val.dev.tsv.tagged.small",
  "model": {
    "type": "crf_tagger",
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "embedding_dim": 256
        }
      }
    },
    "encoder": {
      "type": "gru",
      "input_size": 256,
      "hidden_size": 512,
      "num_layers": 2,
      "dropout": 0.0,
      "bidirectional": true
    },
    "regularizer": [
      ["transitions$", {"type": "l2", "alpha": 0.01}]
    ]
  },
  "iterator": {"type": "basic", "batch_size": 16},
  "trainer": {
    "optimizer": "adam",
    "num_epochs": 50,
    "cuda_device": 0
  },
  "vocabulary": {
    "max_vocab_size": 50000
  }
}
