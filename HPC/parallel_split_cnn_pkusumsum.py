from pathlib import Path
from _collections import defaultdict
import argparse
import shutil
import drmaa
import os
import stat

def split_folders(files, n):
    i = 0
    j = 1
    split_files = defaultdict(list)
    for file in files:
        split_files[j].append(file)
        i += 1
        if i >= n:
            i = 0
            j += 1
    return split_files


def build_tmp_folders(input_path, tmp_path):
    n = args.n
    files = [file for file in input_path.iterdir()]
    split_files = split_folders(files, n)
    tmp_folders = []
    for j, files in split_files.items():
        for file in files:
            (tmp_path / str(j)).mkdir(parents=True, exist_ok=True)
            shutil.copy(str(file), str(tmp_path / str(j) / 'docs' / file.stem))
        tmp_folders.append(tmp_path / str(j))
    return tmp_folders, len(tmp_folders)


def build_sh_scripts(tmp_folders, pkusumsum_path, modules):
    for module, param in modules.items():
        for folder in tmp_folders:
            i = folder.stem
            (folder / module).mkdir(exist_ok=True, parents=True)
            output_str = f'for filename in {str(folder)}/*; do\n'
            output_str += f'\tjava -jar {str(pkusumsum_path)}/PKUSUMSUM.jar -T 1 -input $filename -L 2 -m {param} -n 100 -stop {pkusumsum_path}/lib/stopword_Eng -output {str(folder / module)}/$(basename -- $filename)\n'
            output_str += '\techo $filename\n'
            output_str += 'done'
            file_path = (folder / f'script_{module}.{str(i)}.sh')
            file_path.open('w').write(output_str)
            st = os.stat(str(file_path))
            os.chmod(str(file_path), st.st_mode | stat.S_IEXEC)


def main():
    input_path = Path(args.input_path)
    pkusumsum_path = Path(args.pkusumsum_path)
    tmp_path = Path(args.tmp_path)
    modules = {
        'submodular': '6',
        'centroid': '2',
        'textrank': '5',
        'lexpagerank': '4'
    }
    tmp_folders, num_of_folders = build_tmp_folders(input_path, tmp_path)
    build_sh_scripts(tmp_folders, pkusumsum_path, modules)

    with drmaa.Session() as s:
        for module, _ in modules.items():
            output_str = '#!/usr/bin/env bash\n'
            output_str += f'. {tmp_path}/$SGE_TASK_ID/script_{module}.$SGE_TASK_ID.sh'
            file_path = (tmp_path / f'summarize_{module}.sh')
            file_path.open('w').write(output_str)
            st = os.stat(str(file_path))
            os.chmod(str(file_path), st.st_mode | stat.S_IEXEC)
            jt = s.createJobTemplate()
            jt.blockEmail = False
            jt.joinFiles = True
            jt.workingDirectory = '/home/acp16hh/Salience_Sum'
            jt.email = ['hhardy2@sheffield.ac.uk']
            jt.remoteCommand = file_path
            jt.nativeSpecification = '-l h_rt=48:00:00'
            s.runBulkJobs(jt, 1, num_of_folders, 1)
            s.deleteJobTemplate(jt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-input_path', help='Input path to docs.')
    parser.add_argument('-tmp_path', help='Temporary path for intermediary files.')
    parser.add_argument('-pkusumsum_path', help='PKUSUMSUM path.')
    parser.add_argument('-n', help='Number of parallel process.', type=int)
    args = parser.parse_args()
    main()