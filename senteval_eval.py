from __future__ import absolute_import, division, unicode_literals

import sys
import io
import os
sys.path.append("../SentEval")
# sys.path.append("../../../lirumei/bert_flow/SentEval")
import numpy as np
import logging
import senteval
import json
import argparse

from sentence_transformers import SentenceTransformer, LoggingHandler


logging.basicConfig(format='%(asctime)s - %(filename)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="The saved model path for evaluation")
    parser.add_argument("--main_similarity", type=str, choices=["cosine", "euclidean", "manhattan", "dot_product"], default=None, help="The main similarity type")
    parser.add_argument("--task", type=str, default="STS12,STS13,STS14,STS15,STS16,STSBenchmark,SICKRelatedness", help="The eval task")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--epoch_size", type=int, default=4, help="Epoch")
    parser.add_argument("--nhid", type=int, default=0, help="Hidden layers")
    args = parser.parse_args()
    return args

def prepare(params, samples):
    model = SentenceTransformer(params['model_path'])
    params.model = model

def batcher(params, batch):
    batch = [' '.join(sent) if sent != [] else '.' for sent in batch]
    model = params.model
    embeddings = model.encode(batch)
    return embeddings

def eval_senteval(model_path, transfer_tasks, config):
    # Set params for SentEval
    params = {'task_path': "./data", 'usepytorch': True, 'kfold': 10}
    params['classifier'] = {'nhid': config.nhid, 'optim': 'adam', 'batch_size': config.batch_size,
                                     'tenacity': 5, 'epoch_size': config.epoch_size}
#     params = {'task_path': "./data", 'usepytorch': True, 'kfold': 10}
#     params['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 128,
#                                      'tenacity': 5, 'epoch_size': 2}
    params['model_path'] = model_path
    se = senteval.engine.SE(params, batcher, prepare)
#     transfer_tasks = ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']
    results = se.eval(transfer_tasks)
#     with open(os.path.join(params['model_path'], 'senteval_results.txt'), 'w') as f:
#         f.write(str(results))
    logging.info(f"SentEval results: {results}")
    del results['STSBenchmark']['yhat']
    del results['SICKRelatedness']['yhat']
    json.dump(results, open(os.path.join(params['model_path'], f"senteval_results_h{config.nhid}_e{config.epoch_size}_bsz{config.batch_size}.json"), "w"), indent=4, ensure_ascii=False)
    

if __name__ == "__main__":
    args = parse_args()
    print(">>>", args.task.upper().split(","))
    eval_senteval(args.model_path, args.task.split(","), args)