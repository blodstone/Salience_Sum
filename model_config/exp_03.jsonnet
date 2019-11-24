local HIDDEN=512;
local EMBEDDING=256;
local FFHIDDEN=2048;
local PROJ=256;
{
  "dataset_reader": {
    "type": "summdatareader",
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
  "train_data_path": "data/bbc_allen/train.tsv.tagged",
  "validation_data_path": "data/bbc_allen/val.tsv.tagged",
  "model": {
    "type": "salience_seq2seq",
    "noisy_prediction": {
      "type": "basic_noisy_prediction",
      "hidden_dim": HIDDEN
    },
    "encoder": {
      "type": "stacked_self_attention",
      "input_dim": EMBEDDING,
      "hidden_dim": HIDDEN,
      "projection_dim": PROJ,
      "feedforward_hidden_dim": FFHIDDEN,
      "num_attention_heads": 8,
      "num_layers": 4,
      "dropout_prob": 0.1
    },
    "source_text_embedder": {
      "token_embedders": {
          "tokens": {
              "type": "embedding",
              "embedding_dim": EMBEDDING,
              "trainable": true
          }
      }
    },
    "decoder": {
      "decoder_net": {
         "type": "lstm_cell",
         "decoding_dim": HIDDEN,
         "target_embedding_dim": EMBEDDING
      },
      "max_decoding_steps": 400,
      "target_namespace": "tokens",
      "target_embedder": {
        "vocab_namespace": "tokens",
        "embedding_dim": EMBEDDING
      },
      "scheduled_sampling_ratio": 0.9,
      "beam_size": 5
    }
  },
  "iterator": {
    "type": "bucket",
    "padding_noise": 0.0,
    "batch_size" : 12,
    "instances_per_epoch" : 100000,
    "sorting_keys": [["source_tokens", "num_tokens"]]
  },
  "trainer": {
    "grad_norm": 5.0,
    "grad_clipping": 1.0,
    "summary_interval": 500,
    "histogram_interval": 1000,
    "num_epochs": 50,
    "patience": 10,
    "cuda_device": 0,
    "num_serialized_models_to_keep": 5,
    "optimizer": {
      "type": "adam",
      "lr": 0.01
    },
  },
  "vocabulary": {
    "max_vocab_size": 50000
  }
}
