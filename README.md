# SalienceSum

## Installation
```bash
python3 -m venv venv
pip install -r requirements.txt
```

## Data
We use the BBC data that comprised of: 
- `train_src.txt` and `train_tgt.txt`
- `val_src.txt` and `val_tgt.txt`
- `test_src.txt` and `test_tgt.txt`

There's another separate set of test data that

## Generating noisy models
To generate an unsupervised noisy salience, we process all set of source files.

```bash
python preprocess-noisy.py -input <path_to_source_folder> --submodular --NER --textrank --compression -max-words 30
```

The process will generate labeled source files using `||` separator as follows.

```text
token1||salience1 token2||salience2
```

