local HIDDEN=100;
local EMBEDDING=50;
local PROJECTION=25;
local FFHIDDEN=25;

{
  "dataset_reader": {
    "type": "summdatareader",
    "tokenizer": {
      "type": "word",
      "word_splitter": {
        "type": "just_spaces"
      }
    },
    "source_max_tokens": 400
  },
  "train_data_path": "../data/dev_bbc/train.dev.tsv.tagged.small",
  "validation_data_path": "../data/dev_bbc/val.dev.tsv.tagged.small",
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
        "projection_dim": PROJECTION,
        "feedforward_hidden_dim": FFHIDDEN,
        "num_attention_heads": 3,
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
    "batch_size" : 8,
    "sorting_keys": [["source_tokens", "num_tokens"]]
  },
  "trainer": {
    "histogram_interval": 1,
    "num_epochs": 4,
    "patience": 10,
    "cuda_device": -1,
    "grad_norm": 2,
    "optimizer": {
      "type": "adagrad",
      "lr": 0.15,
      "initial_accumulator_value": 0.1
    },
  },
  "vocabulary": {
    "max_vocab_size": 10
  }
}
