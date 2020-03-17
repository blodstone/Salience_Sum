# -*- encoding: utf-8 -*-
import argparse
import os
import time
from pathlib import Path
from multiprocessing import Pool
import pyrouge
import shutil
import sys
import codecs


def rouge(file_input):
    """Calculate ROUGE scores of sequences passed as an iterator
       e.g. a list of str, an open file, StringIO or even sys.stdin
    """
    name, ref, cand = file_input
    current_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
    tmp_dir = ".rouge-tmp-{}-{}".format(current_time, name)
    try:
        if not os.path.isdir(tmp_dir):
            os.mkdir(tmp_dir)
            os.mkdir(tmp_dir + "/candidate")
            os.mkdir(tmp_dir + "/reference")
        candidates = [line.strip() for line in cand if line.strip() != '']
        references = [line.strip().split('\t')[1] for line in ref if line.strip() != '']
        assert len(candidates) == len(references), f'{len(candidates)}, {len(references)}'
        cnt = len(candidates)
        for i in range(cnt):
            if len(references[i]) < 1:
                continue
            with open(tmp_dir + "/candidate/cand.{}.txt".format(i), "w",
                      encoding="utf-8") as f:
                f.write(candidates[i])
            with open(tmp_dir + "/reference/ref.{}.txt".format(i), "w",
                      encoding="utf-8") as f:
                f.write(references[i])
        r = pyrouge.Rouge155(rouge_dir='/home/acp16hh/Projects/Others/ROUGE')
        r.model_dir = tmp_dir + "/reference/"
        r.system_dir = tmp_dir + "/candidate/"
        r.model_filename_pattern = 'ref.#ID#.txt'
        r.system_filename_pattern = r'cand.(\d+).txt'
        rouge_args = '-e /home/acp16hh/Projects/Others/ROUGE/data -c 95 -2 -1 -U -r 1000 -n 2 -w 1.2 -a -m -d'
        # rouge_results = r.convert_and_evaluate(rouge_args=rouge_args)
        rouge_results = r.convert_and_evaluate()
        # print(rouge_results)
        results_dict = r.output_to_dict(rouge_results)
        results_dict['name'] = name
        return results_dict
    except:
        return None
    finally:
        pass
        if os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir)


def rouge_results_to_str(results_dict):
    return "{},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f}\n".format(
        results_dict['name'],
        results_dict["rouge_1_f_score"] * 100,
        results_dict["rouge_2_f_score"] * 100,
        results_dict["rouge_3_f_score"] * 100,
        results_dict["rouge_l_f_score"] * 100,
        results_dict["rouge_su*_f_score"] * 100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', help='folder containing candidate files.', default='')
    parser.add_argument('-c', type=str, default="candidate.txt",
                        help='candidate file (TXT format)')
    parser.add_argument('-r', type=str, default="reference.tsv",
                        help='reference file (TSV format)')
    parser.add_argument('-o', type=str, help='output path.')
    parser.add_argument('-n', type=str, help='output filename.')
    args = parser.parse_args()

    results = []
    if args.f == '':
        if args.c.upper() == "STDIN":
            candidates = sys.stdin
        else:
            candidates = codecs.open(args.c, encoding="utf-8")
        references = codecs.open(args.r, encoding="utf-8")
        results.append(rouge((Path(args.f).stem, references, candidates)))
    else:
        input_folder = Path(args.f)
        inputs = []
        for f in input_folder.iterdir():
            inputs.append((f.stem,
                           codecs.open(args.r, encoding="utf-8").readlines(),
                           f.open(encoding='utf-8').readlines()))
        with Pool(16) as p:
            results = p.map(rouge, inputs)

    output_path = Path(args.o)
    (output_path / args.n).open('w').write('')
    (output_path / args.n).open('a').write('Name,R1,R2,R3,RL,RSU4\n')
    for result in results:
        if result is not None:
            (output_path / args.n).open('a').write(rouge_results_to_str(result))
