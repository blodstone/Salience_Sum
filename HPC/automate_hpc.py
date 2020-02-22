#  Copyright (c) Hardy (hardy.oei@gmail.com) 2020.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import os
import stat
from pathlib import Path

import drmaa
import argparse


def create_sh(mode):
    spec_file = Path(args.spec_file)
    output_path = spec_file.absolute().parents[0]
    output_path.mkdir(parents=True, exist_ok=True)
    server_output = {
        'sharc': [],
        'dgx': [],
    }
    seeds = {
        'sharc': [],
        'dgx': [],
    }
    # Eg:pg_salience_feature,pg_salience_emb_mlp_16.jsonnet,emb_mlp_16,dgx,bbc
    for i, line in enumerate(spec_file.open().readlines()):
        package, jsonnet, model, server, dataset, random_seed, numpy_seed, pytorch_seed = line.split(',')
        if package == 'pg_salience_feature':
            test = 'test.salience.tsv'
        else:
            test = 'test.tsv'
        output_str = f'MODEL=/data/acp16hh/Exp_Gwen_Saliency/{package}/{dataset}/{server}/{model}\n'
        output_str += f'DATA=/data/acp16hh/data/{dataset}\n'
        output_str += 'module load apps/python/conda\n'
        output_str += 'module load libs/cudnn/7.3.1.20/binary-cuda-9.0.176\n'
        output_str += 'source activate gwen\n'
        output_str += f'allennlp train -s $MODEL -f ' \
                      f'--file-friendly-logging ' \
                      f'--include-package {package} ' \
                      f'/home/acp16hh/Salience_Sum/HPC/{package}/{jsonnet}\n'
        output_str += f'python /home/acp16hh/Salience_Sum/postprocess/retrieve_last_model.py $MODEL\n'
        output_str += f'python /home/acp16hh/Salience_Sum/summarize.py -module {package} ' \
                      f'-input $DATA/ready/{test} -vocab_path $MODEL/vocabulary ' \
                      f'-model $MODEL/pick.th ' \
                      f'-model_config /home/acp16hh/Salience_Sum/HPC/{package}/{jsonnet} ' \
                      f'-output_path ' \
                      f'$DATA/result/test_{package}_{dataset}_{server}_{model}.out -batch_size 24'
        seed_str = f'export RANDOM_SEED={random_seed}\n' \
                   f'export NUMPY_SEED={numpy_seed}\n' \
                   f'export PYTORCH_SEED={pytorch_seed}'
        if server == 'sharc':
            server_output['sharc'].append(output_str)
            seeds['sharc'].append(seed_str)
        elif server == 'dgx':
            server_output['dgx'].append(output_str)
            seeds['dgx'].append(seed_str)
    if mode == 'sharc' or mode == 'all':
        for i, output_str in enumerate(server_output['sharc']):
            seed_str = seeds['sharc'][i]
            (output_path / 'sharc').mkdir(parents=True, exist_ok=True)
            file_path = (output_path / 'sharc' / f'script_{str(i+1)}.sh')
            file_path.write_text(f'#!/bin/bash\n{seed_str}\n{output_str}')
            st = os.stat(str(file_path))
            os.chmod(str(file_path), st.st_mode | stat.S_IEXEC)
        job_file = (output_path / 'jobs_sharc.sh')
        job_file.write_text(f'#!/bin/bash\n. {str(output_path)}/sharc/script_$SGE_TASK_ID.sh')
        st = os.stat(str(job_file))
        os.chmod(str(job_file), st.st_mode | stat.S_IEXEC)
    if mode == 'dgx' or mode == 'all':
        for i, output_str in enumerate(server_output['dgx']):
            seed_str = seeds['dgx'][i]
            (output_path / 'dgx').mkdir(parents=True, exist_ok=True)
            file_path = (output_path / 'dgx' / f'script_{str(i+1)}.sh')
            file_path.write_text(f'#!/bin/bash\n{seed_str}\n{output_str}')
            st = os.stat(str(file_path))
            os.chmod(str(file_path), st.st_mode | stat.S_IEXEC)
        job_file = (output_path / 'jobs_dgx.sh')
        job_file.write_text(f'#!/bin/bash\n. {str(output_path)}/dgx/script_$SGE_TASK_ID.sh')
        st = os.stat(str(job_file))
        os.chmod(str(job_file), st.st_mode | stat.S_IEXEC)
    return server_output


def main():
    if args.sharc:
        mode = 'sharc'
    elif args.dgx:
        mode = 'dgx'
    else:
        mode = 'all'
    server_output = create_sh(mode)
    with drmaa.Session() as s:
        if mode == 'sharc' or mode == 'all':
            build_run_job(s, 'sharc', len(server_output['sharc']))
        if mode == 'dgx' or mode == 'all':
            build_run_job(s, 'dgx', len(server_output['dgx']))


def build_run_job(s, mode, last_i):
    output_path = Path(args.spec_file).absolute().parents[0]
    jt = s.createJobTemplate()
    jt.blockEmail = False
    jt.joinFiles = True
    jt.workingDirectory = '/home/acp16hh/Salience_Sum'
    jt.email = ['hhardy2@sheffield.ac.uk']
    jt.remoteCommand = str(output_path / f'jobs_{mode}.sh')
    if mode == 'sharc':
        jt.nativeSpecification = '-P gpu -l gpu=1 -tc 5 -l rmem=48G -l h_rt=96:00:00'
    elif mode == 'dgx':
        jt.nativeSpecification = '-P rse -q rse.q -tc 2 -l gpu=1 -l rmem=48G -l h_rt=96:00:00'
    s.runBulkJobs(jt, 1, last_i, 1)
    s.deleteJobTemplate(jt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sharc', help='Run script for sharc only.', action='store_true')
    parser.add_argument('--dgx', help='Run script for dgx only.', action='store_true')
    parser.add_argument('-spec_file', help='Path to spec file.')
    args = parser.parse_args()
    main()
