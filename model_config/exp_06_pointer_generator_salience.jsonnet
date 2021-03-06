local HIDDEN=256;
local EMBEDDING=128;
local CUDA=0;
{
  "dataset_reader": {
    "type": "summdatareader_salience",
    "lazy": false,
    "interpolation": false,
    "predict": false,
    "use_salience": true,
    "source_max_tokens": 400,
    "target_max_tokens": 100,
  },
  "train_data_path": "data/bbc_allen/train.tsv.tagged",
  "validation_data_path": "data/bbc_allen/validation.tsv.tagged",
  "model": {
    "type": "encoder_decoder_salience",
    "encoder": {
      "input_size": EMBEDDING,
      "hidden_size": HIDDEN,
      "num_layers": 1,
      "bidirectional": true
    },
    "teacher_force_ratio": 0.8,
    "coverage_lambda": 0.0,
    "salience_lambda": 0.0,
    "decoder": {
      "attention": {
        "hidden_size": HIDDEN,
        "bidirectional": true,
      },
      "input_size": EMBEDDING,
      "hidden_size": HIDDEN,
      "num_layers": 1,
    },
    "salience_predictor": {
      "hidden_size": HIDDEN,
      "bidirectional": true,
    },
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
    "summary_interval": 500,
    "histogram_interval": 1000,
    "num_epochs": 50,
    "patience": 5,
    "cuda_device": CUDA,
    "num_serialized_models_to_keep": 5,
    "grad_clipping": 2,
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
