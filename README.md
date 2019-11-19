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
python preprocess-salience.py -input <path_to_source_folder> --submodular --NER --textrank --compression -max-words 30
```

The process will generate labeled source files using `||` separator as follows.

```text
token1||salience1 token2||salience2
```

## Current preprocess setting (for development purpose)
Extract from raw data into data suitable for PKUSUMSUM
```bash
python noisy_salience_model/preprocess_submodular.py
```
The noisy models are using the unsupervised models by PKUSUMSUM system. There are three that are in used: textrank, centroid and submodular. 

Run the PKUSUMSUM system first and generate each of the model summary into their respective folder.Then run a script for preprocessing the result into a text file. The doc path is fixed (with exception of the system's summ path) but the output path has to be manually changed for each system. This is for temporary measure only.
```bash
python special_preprocess.py
```
Convert all the documents and summaries to spacy object. For each PKUSUMSUM output, we have to change to it respective output.
```bash
python preprocess-spacy.py -tgt sample_data/textrank_train_tgt.txt -output sample_data
```

```bash
python preprocess-salience.py -src sample_data/train_src.pickle --AKE --submodular -submodular_tgt sample_data/submodular_train_tgt.pickle --centroid -centroid_tgt sample_data/centroid_train_tgt.pickle --textrank -textrank_tgt sample_data/textrank_train_tgt.pickle --NER -max_words 30 --gold -highlight sample_data/df_gold.pickle -doc_id sample_data/doc_id.txt
```

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
python preprocess/add_noisy_dummy.pu
```
