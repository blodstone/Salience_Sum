# SalienceSum

## Installation
```bash
python3 -m venv venv
pip install -r requirements.txt
```

## Data
We use the BBC data that comprised of the raw version: 
- `train.tsv`
- `test.tsv`
- `val.tsv`
and the salience model tagged version:
- `train.tsv.tagged`
- `test.tsv.tagged`
- `val.tsv.tagged`

## Tagging the document using multiple noisy models
### PKUSUM
Split the one file document into document files for PKUSUM processing.
*Need to argparse the path, for now we change the path manually*
```bash
cd preprocess
python generate_docs_pkusum.py
```
Run the script. Each script for training takes 30 hours to run (the val will take ~3 hours).  
```bash
./centroid.sh
./centroid_val.sh
./submodular.sh
./submodular_val.sh
./textrank.sh
./textrank_val.sh
```
Once all scripts are done, run
```bash
python preprocess_salience_from_pkupu.py
```

```bash
allennlp train model_config/exp_01.jsonnet --serialization-dir data/train_01 --include-package salience_sum --file-friendly-logging
```
## Generating noisy models
To generate an unsupervised noisy salience, we process all set of source files.

```bash
python preprocess-salience.py -input <path_to_source_folder> --submodular --NER --textrank --compression -max-words 30
```

The process will generate labeled source files using `|#|` separator as follows.


# Salience AutoEncoder

Replace manually the file name inside the preprocessing.
```bash
python preprocess/preprocessing.py
```

Create a smaller one for dev.
```bash
cat data/bbc/train.tsv | awk 'NR%200==1' > data/dev_bbc/train.dev.tsv
cat data/bbc/val.tsv | awk 'NR%100==1' > data/dev_bbc/val.dev.tsv
```

```bash
python preprocess/add_noisy_dummy.py
```

