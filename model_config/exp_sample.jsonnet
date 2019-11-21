local HIDDEN=256;
local EMBEDDING=128;
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
  "train_data_path": "data/dev_bbc/train.dev.tsv.tagged.small",
  "validation_data_path": "data/dev_bbc/val.dev.tsv.tagged.small",
  "model": {
    "type": "salience_seq2seq",
    "noisy_prediction": {
      "type": "basic_noisy_prediction",
      "hidden_dim": HIDDEN
    },
    "encoder": {
      "type": "seq2seqwrapper",
      "module": {
        "type": "denoising_encoder",
        "bidirectional": true,
        "num_layers": 2,
        "use_bridge": true,
        "input_size": EMBEDDING,
        "hidden_size": HIDDEN
      },
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
    "grad_norm": 2.0,
    "histogram_interval": 1,
    "num_epochs": 50,
    "patience": 10,
    "cuda_device": -1,
    "optimizer": {
      "type": "adagrad",
      "lr": 0.15,
      "initial_accumulator_value": 0.1
    },

  }
}
