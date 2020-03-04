import argparse
import re
import shutil
from pathlib import Path


def main():
    model_folder = Path(args.model_folder)
    p = re.compile(r'(?P<epoch>[0-9]+)\.th')
    epoch = -1
    for file in model_folder.iterdir():
        m = p.search(file.name)
        if m is not None:
            if int(m.group('epoch')) > epoch:
                epoch = int(m.group('epoch'))
    if epoch != -1:
        model = f'model_state_epoch_{str(epoch)}'
        print(f'Copying model {model} as pick.th.')
        shutil.copy(str(model_folder / model), str(model_folder / 'pick.th'))
    else:
        print('No model state found.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_folder', help='The model folder.')
    args = parser.parse_args()
    main()
