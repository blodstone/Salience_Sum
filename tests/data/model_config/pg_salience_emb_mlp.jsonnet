local HIDDEN=20;
local EMBEDDING=10;
local FEATURE=3;
local CUDA=0;
{
  "random_seed": 100,
  "numpy_seed": 100,
  "pytorch_seed": 100,
  "dataset_reader": {
    "type": "summdatareader_salience_feature",
    "lazy": false,
    "interpolation": false,
    "predict": false,
    "use_salience": true,
    "source_max_tokens": 400,
    "target_max_tokens": 100,
  },
  "train_data_path": '',
  "validation_data_path": '',
  "model": {
    "type": "enc_dec_salience_feature",
    "encoder": {
      "input_size": EMBEDDING,
      "hidden_size": HIDDEN,
      "num_layers": 1,
      "bidirectional": true
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
    "teacher_force_ratio": 0.7,
    "decoder": {
      "attention": {
        "hidden_size": HIDDEN,
        "bidirectional": true,
      },
      "input_size": EMBEDDING,
      "hidden_size": HIDDEN,
      "is_emb_attention": true,
      "emb_attention_mode": "mlp",
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
    "batch_size" : 2,
    "sorting_keys": [["source_tokens", "num_tokens"]]
  },
  "trainer": {
    "summary_interval": 1000,
    "histogram_interval": 1000,
    "num_epochs": 2,
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
