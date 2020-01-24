#  Copyright (c) Hardy (hardy.oei@gmail.com) 2020.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from typing import List, Union

from spacy.tokens import Doc

from noisy_salience_model.salience_model import Instance

Text = List[Union[List[str], Doc]]


def test_build_map():
    input_raw = [['This', 'is', '1'], ['This', 'is', '2']]
    index_map = {
        0: (0, 0),
        1: (0, 1),
        2: (0, 2),
        3: (1, 0),
        4: (1, 1),
        5: (1, 2),
    }
    input_doc = [['this', 'is', '1'], ['this', 'is', '2']]
    input_summ = [['This', 'is', '2']]
    instance = Instance(doc=input_doc, raw=input_raw, summ=input_summ)
    for index in instance.index_map.keys():
        assert instance.index_map[index] == index_map[index], \
            f'At {index} expected {index_map[index]}, got {instance.index_map[index]}'