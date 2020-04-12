local HIDDEN=20;
local EMBEDDING=10;
local FEATURE=3;
local CUDA=-1;
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
  "model": {
    "type": "enc_dec_salience_feature",
    "encoder": {
      "input_size": EMBEDDING,
      "hidden_size": HIDDEN,
      "num_layers": 1,
      "bidirectional": true
    },
    "teacher_force_ratio": 1.0,
    "salience_source_mixer":{
      "type": "bilinear_attn",
      "embedding_size": EMBEDDING,
      "feature_size": FEATURE,
      "k_size": 10,
      "c_size": 10,
      "p_size": 10,
      "glimpse" : 4,
      "salience_embedder": {
        "embedding_size": EMBEDDING,
        "feature_size": FEATURE,
        "type": "matrix",
      },
    },
    "decoder": {
      "attention": {
        "hidden_size": HIDDEN,
        "bidirectional": true,
      },
      "input_size": EMBEDDING,
      "hidden_size": HIDDEN,
    },
    "coverage_lambda": 0.1,
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
    "batch_size" : 2,
    "sorting_keys": [["source_tokens", "num_tokens"]]
  },
  "trainer": {
    "summary_interval": 100,
    "histogram_interval": 100,
    "num_epochs": 3,
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
    "max_vocab_size": 2000
  }
}