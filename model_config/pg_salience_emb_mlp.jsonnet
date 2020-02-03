local HIDDEN=512;
local EMBEDDING=128;
local FEATURE=6;
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
  "train_data_path": "/home/acp16hh/Projects/Research/Experiments/Exp_Gwen_Saliency/src/SalienceSum/data/bbc/ready/all/train.concat.tsv",
  "validation_data_path": "/home/acp16hh/Projects/Research/Experiments/Exp_Gwen_Saliency/src/SalienceSum/data/bbc/ready/all/validation.concat.tsv",
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
    "patience": 8,
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
