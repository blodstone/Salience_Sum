#  Copyright (c) Hardy (hardy.oei@gmail.com) 2020.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import time
import sys

n = sys.argv[1]
time.sleep(30)
open(f'simple_{n}.csv').write(f'{n}')