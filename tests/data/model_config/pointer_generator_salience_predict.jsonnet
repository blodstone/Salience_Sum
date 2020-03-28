local HIDDEN=512;
local EMBEDDING=128;
local FEATURE=6;
local CUDA=-1;
{
  "dataset_reader": {
    "type": "summdatareader_salience",
    "predict": false,
    "lazy": true,
    "interpolation": false,
    "source_max_tokens": 400
  },
  "test_data_path": "data/bbc_allen/val.tsv.tagged.small",
  "model": {
    "type": "encoder_decoder_salience",
    "encoder": {
      "input_size": EMBEDDING,
      "hidden_size": HIDDEN,
      "num_layers": 1,
      "bidirectional": true
    },
    "cuda": CUDA,
    "decoder": {
      "attention": {
        "input_size": EMBEDDING,
        "hidden_size": HIDDEN,
        "bidirectional": true,
      },
      "input_size": EMBEDDING,
      "hidden_size": HIDDEN,
      "num_layers": 1,
    },
    "hidden_size": HIDDEN,
    "max_steps": 100,
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
    "batch_size" : 8,
    "sorting_keys": [["source_tokens", "num_tokens"]]
  },
  "trainer": {
    "summary_interval": 1,
    "histogram_interval": 1,
    "num_epochs": 2,
    "patience": 1,
    "cuda_device": CUDA,
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
