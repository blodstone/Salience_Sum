local HIDDEN=256;
local EMBEDDING=128;
{
  "dataset_reader": {
    "type": "summdatareader",
    "lazy": false,
    "interpolation": false,
    "source_max_tokens": 400
  },
  "train_data_path": "data/bbc_allen/train.tsv.tagged",
  "validation_data_path": "data/bbc_allen/val.tsv.tagged",
  "model": {
    "type": "encoder_decoder",
    "max_target_size": 100,
    "beam_size": 5,
    "encoder": {
      "input_size": EMBEDDING,
      "hidden_size": HIDDEN,
      "num_layers": 1,
      "bidirectional": true
    },
    "decoder": {
      "attention": {
        "hidden_size": HIDDEN,
      },
      "input_size": EMBEDDING,
      "hidden_size": HIDDEN,
      "num_layers": 1,
    },
    "hidden_size": HIDDEN,
    "embedder": {
      "token_embedders": {
          "tokens": {
              "type": "embedding",
              "embedding_dim": EMBEDDING,
              "trainable": true
          }
      }
    },
  },
  "iterator": {
    "type": "bucket",
    "padding_noise": 0.0,
    "batch_size" : 16,
    "sorting_keys": [["source_tokens", "num_tokens"]]
  },
  "trainer": {
    "summary_interval": 1,
    "histogram_interval": 1,
    "num_epochs": 2,
    "patience": 1,
    "cuda_device": -1,
    "num_serialized_models_to_keep": 1,
    "grad_norm": 2,
    "optimizer": {
      "type": "adagrad",
      "lr": 0.15,
      "initial_accumulator_value": 0.1
    },
  },
  "vocabulary": {
    "max_vocab_size": 50000
  }
}
