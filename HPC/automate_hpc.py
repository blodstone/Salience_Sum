#  Copyright (c) Hardy (hardy.oei@gmail.com) 2020.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import os
import stat
from pathlib import Path

import drmaa
import argparse


def create_sh(mode, param):
    """
    Creates all the scripts that would be run by the job script.
    :param mode: sharc or dgx
    :return: dictionary containing scripts with mode as keys
    """
    spec_file = Path(args.spec_file)
    data_path = Path(args.data_path)
    output_path = spec_file.absolute().parents[0]
    script_name = spec_file.absolute().parents[0].stem
    output_path.mkdir(parents=True, exist_ok=True)
    server_output = {
        'sharc': [],
        'dgx': [],
    }
    seeds = {
        'sharc': [],
        'dgx': [],
    }
    # Eg:pg_salience_feature,pg_salience_emb_mlp_16.jsonnet,emb_mlp_16,dgx,bbc,seed
    for i, line in enumerate(spec_file.open().readlines()):
        if line.strip() == '':
            continue
        package, jsonnet, model, server, dataset, seed = line.split(',')
        use_salience = '--use_salience'
        if 'clean' in jsonnet:
            use_salience = ''
        seed = seed.strip()
        train_path = data_path / dataset / 'ready' / 'train.salience.tsv'
        test_path = data_path / dataset / 'ready' / 'test.salience.tsv'
        validation_path = data_path / dataset / 'ready' / 'validation.salience.tsv'
        json_path = f'/home/acp16hh/Salience_Sum/HPC/{package}/{jsonnet}'
        if param == '-r':
            json_path = f'/data/acp16hh/Exp_Gwen_Saliency/{package}/{dataset}/{server}/{model}/{seed}/config.json'
        output_str = f'MODEL=/data/acp16hh/Exp_Gwen_Saliency/{package}/{dataset}/{server}/{model}/{seed}\n'
        output_str += f'DATA={str(data_path)}/{dataset}\n'
        output_str += f'OUTPUT={str(data_path)}/result\n'
        output_str += 'module load apps/python/conda\n'
        output_str += 'module libs/cudnn/7.6.5.32/binary-cuda-10.0.130\n'
        output_str += 'source activate gwen\n'
        output_str += f'export train_path={str(train_path)}\n'
        output_str += f'export validation_path={str(validation_path)}\n'
        if not args.summarizer_only:
            output_str += f'allennlp train -s $MODEL {param} ' \
                          f'--file-friendly-logging ' \
                          f'--include-package {package} ' \
                          f'{json_path}\n'
        if args.last_summary:
            output_str += f'python /home/acp16hh/Salience_Sum/postprocess/retrieve_last_model.py $MODEL\n'
            output_str += f'python /home/acp16hh/Salience_Sum/summarize.py ' \
                          f'-input {str(test_path)} -vocab_path $MODEL/vocabulary ' \
                          f'-model $MODEL/pick.th ' \
                          f'-model_config /home/acp16hh/Salience_Sum/HPC/{package}/{jsonnet} ' \
                          f'-output_path ' \
                          f'$DATA/result/test_{package}_{dataset}_{server}_{model}_{seed}_last.out -batch_size 48 ' \
                          f'--cuda {use_salience}\n '
        if args.best_summary:
            output_str += f'python /home/acp16hh/Salience_Sum/summarize.py ' \
                          f'-input {str(test_path)} -vocab_path $MODEL/vocabulary ' \
                          f'-model $MODEL/best.th ' \
                          f'-model_config /home/acp16hh/Salience_Sum/HPC/{package}/{jsonnet} ' \
                          f'-output_path ' \
                          f'$DATA/result/test_{package}_{dataset}_{server}_{model}_{seed}_best.out -batch_size 48 ' \
                          f'--cuda {use_salience}\n '
        seed_str = f'export RANDOM_SEED={seed}\n' \
                   f'export NUMPY_SEED={seed}\n' \
                   f'export PYTORCH_SEED={seed}'
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
        job_file = (output_path / f'jobs_{script_name}_sharc.sh')
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
        job_file = (output_path / f'jobs_{script_name}_dgx.sh')
        job_file.write_text(f'#!/bin/bash\n. {str(output_path)}/dgx/script_$SGE_TASK_ID.sh')
        st = os.stat(str(job_file))
        os.chmod(str(job_file), st.st_mode | stat.S_IEXEC)
    return server_output


def main():
    if args.force:
        param = '-f'
    else:
        param = '-r'
    if args.sharc:
        mode = 'sharc'
    elif args.dgx:
        mode = 'dgx'
    else:
        mode = 'all'
    server_output = create_sh(mode, param)
    with drmaa.Session() as s:
        if mode == 'sharc' or mode == 'all':
            build_run_job(s, 'sharc', len(server_output['sharc']))
        if mode == 'dgx' or mode == 'all':
            build_run_job(s, 'dgx', len(server_output['dgx']))


def build_run_job(s, mode, last_i):
    output_path = Path(args.spec_file).absolute().parents[0]
    script_name = Path(args.spec_file).absolute().parents[0].stem
    jt = s.createJobTemplate()
    jt.blockEmail = False
    jt.joinFiles = True
    jt.workingDirectory = '/home/acp16hh/Salience_Sum'
    jt.email = ['hhardy2@sheffield.ac.uk']
    jt.remoteCommand = str(output_path / f'jobs_{script_name}_{mode}.sh')
    if mode == 'sharc':
        jt.nativeSpecification = '-P gpu -l gpu=1 -tc 5 -l rmem=36G -l h_rt=72:00:00'
    elif mode == 'dgx':
        jt.nativeSpecification = '-P rse -q rse.q -tc 5 -l gpu=1 -l rmem=36G -l h_rt=72:00:00'
    s.runBulkJobs(jt, 1, last_i, 1)
    s.deleteJobTemplate(jt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sharc', help='Run script for sharc only.', action='store_true')
    parser.add_argument('--dgx', help='Run script for dgx only.', action='store_true')
    parser.add_argument('--force', help='Apply -f to allennlp training. Will delete the data.', action='store_true')
    parser.add_argument('--best_summary', help='Run best.th summarizer.', action='store_true')
    parser.add_argument('--last_summary', help='Run last model summarizer.', action='store_true')
    parser.add_argument('--summarizer_only', help='Only run summarizer.', action='store_true')
    parser.add_argument('-data_path', help='Data path.')
    parser.add_argument('-spec_file', help='Path to spec file.')
    args = parser.parse_args()
    main()
