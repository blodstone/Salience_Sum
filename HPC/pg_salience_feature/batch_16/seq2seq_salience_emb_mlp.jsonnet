local HIDDEN=512;
local EMBEDDING=128;
local FEATURE=6;
local CUDA=0;
{
  "random_seed": std.parseInt(std.extVar("RANDOM_SEED")),
  "numpy_seed": std.parseInt(std.extVar("NUMPY_SEED")),
  "pytorch_seed": std.parseInt(std.extVar("PYTORCH_SEED")),
  "dataset_reader": {
    "type": "summdatareader_salience_feature",
    "lazy": false,
    "interpolation": false,
    "predict": false,
    "use_salience": true,
    "source_max_tokens": 400,
    "target_max_tokens": 100,
  },
  "train_data_path": std.extVar("train_path"),
  "validation_data_path": std.extVar("validation_path"),
  "model": {
    "type": "enc_dec_salience_feature",
    "encoder": {
      "input_size": EMBEDDING,
      "hidden_size": HIDDEN,
      "num_layers": 1,
      "bidirectional": true
    },
    "use_copy_mechanism": false,
    "teacher_force_ratio": 0.7,
    "decoder": {
      "attention": {
        "hidden_size": HIDDEN,
        "bidirectional": true
      },
      "input_size": EMBEDDING,
      "hidden_size": HIDDEN,
      "use_copy_mechanism": false,
      "is_emb_attention": false,
      "emb_attention_mode": "mlp",
    },
    "salience_source_mixer":{
      "type": "emb_mlp",
      "embedding_size": EMBEDDING,
      "feature_size": FEATURE,
      "hidden_size": HIDDEN,
      "salience_embedder": {
        "embedding_size": EMBEDDING,
        "feature_size": FEATURE,
        "type": "vector",
      },
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
    "summary_interval": 1000,
    "histogram_interval": 1000,
    "num_epochs": 30,
    "patience": 5,
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
