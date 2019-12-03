import logging

from allennlp.common.testing import ModelTestCase


class SalienceModelTest(ModelTestCase):

    def setUp(self):
        super().setUp()
        logging.basicConfig(level=logging.INFO)
        self.set_up_model(
            '../model_config/exp_debug.jsonnet',
            '../data/dev_bbc/train.dev.tsv.tagged')

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)
