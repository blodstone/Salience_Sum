from pathlib import Path

from typing import Callable, Dict

from tests.conftest import ModelTestCase
from pointer_generator_salience.reader.summ_reader import SummDataReader


class TestPointerGeneratorSalienceModel:

    def test_run(self, model: Callable[[Path, Dict, Path], ModelTestCase],
                   shared_datadir: Path, datadir: Path):
        param_file = shared_datadir / 'model_config' / 'pointer_generator_salience.jsonnet'
        dataset_file = {
            'train': '/home/acp16hh/Projects/Research/Experiments/Exp_Gwen_Saliency_Summ/src/Salience_Sum/data/bbc_dev/ready/train.sum.tsv',
            'validation': '/home/acp16hh/Projects/Research/Experiments/Exp_Gwen_Saliency_Summ/src/Salience_Sum/data/bbc_dev/ready/validation.sum.tsv'
        }
        my_model = model(param_file, dataset_file, datadir)
        my_model.ensure_model_can_train_save_and_load(my_model.param_file)

