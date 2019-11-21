local HIDDEN=512;
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
  "train_data_path": "data/bbc_allen/train.tsv.tagged",
  "validation_data_path": "data/bbc_allen/val.tsv.tagged",
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
        "num_layers": 1,
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
    "batch_size" : 16,
    "sorting_keys": [["source_tokens", "num_tokens"]]
  },
  "trainer": {
    "histogram_interval": 1,
    "num_epochs": 50,
    "patience": 10,
    "cuda_device": -1,
    "grad_norm": 2,
    "optimizer": {
      "type": "ada",
      "lr": 0.15,
      "initial_accumulator_value": 0.1
    },

  }
}
