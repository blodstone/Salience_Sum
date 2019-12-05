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
          "embedding_dim": 50
        }
      }
    },
    "encoder": {
      "type": "gru",
      "input_size": 50,
      "hidden_size": 300,
      "num_layers": 2,
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
    "num_epochs": 5,
    "cuda_device": -1
  },
  "vocabulary": {
    "max_vocab_size": 50000
  }
}
