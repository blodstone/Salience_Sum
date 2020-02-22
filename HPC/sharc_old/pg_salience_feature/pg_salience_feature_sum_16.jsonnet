local HIDDEN=512;
local EMBEDDING=128;
local FEATURE=1;
local CUDA=0;
{
  "dataset_reader": {
    "type": "summdatareader_salience_feature",
    "lazy": false,
    "interpolation": false,
    "predict": false,
    "use_salience": true,
    "source_max_tokens": 400,
    "target_max_tokens": 100,
  },
  "train_data_path": "/data/acp16hh/data/bbc/ready/all/train.sum.tsv",
  "validation_data_path": "/data/acp16hh/data/bbc/ready/all/validation.sum.tsv",
  "model": {
    "type": "enc_dec_salience_feature",
    "encoder": {
      "input_size": EMBEDDING + FEATURE,
      "hidden_size": HIDDEN,
      "num_layers": 1,
      "bidirectional": true
    },
    "teacher_force_ratio": 1.0,
    "decoder": {
      "attention": {
        "hidden_size": HIDDEN,
        "bidirectional": true,
      },
      "input_size": EMBEDDING,
      "hidden_size": HIDDEN,
    },
    "coverage_lambda": 0.0,
    "max_steps": 100,
    "source_embedder": {
      "token_embedders": {
          "tokens": {
              "type": "embedding",
              "embedding_dim": EMBEDDING,
              "trainable": true
          }
      }
    },
    "target_embedder": {
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
    "summary_interval": 100,
    "histogram_interval": 100,
    "num_epochs": 50,
    "cuda_device": CUDA,
    "num_serialized_models_to_keep": 5,
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
