import logging

from allennlp.common.testing import ModelTestCase


class SalienceModelTest(ModelTestCase):

    def setUp(self):
        super().setUp()
        logging.basicConfig(level=logging.INFO)
        self.set_up_model(
            '/home/acp16hh/Projects/Research/Experiments/Exp_Gwen_Saliency_Summ/src/Salience_Sum/model_config/exp_05_debug_crf.jsonnet',
            '../data/dev_bbc/train.dev.tsv.tagged')

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)
