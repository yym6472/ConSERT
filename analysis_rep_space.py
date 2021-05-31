import gzip
import csv
import argparse
import json

from sentence_transformers import models, losses
from sentence_transformers import SentencesDataset, LoggingHandler, SentenceTransformer, util, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, SimilarityFunction
from eval import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./final_output/bert-base-uncased", help="The saved model path")
    parser.add_argument("--output_dir", type=str, default="./tmp/bert-base-uncased", help="The output dir")
    parser.add_argument("--filter_by", type=str, default="freq", choices=["freq", "tfidf"], help="Use which metric to filter token ids")
    parser.add_argument("--num_filter_freq_rank_leq_than", type=int, default=50)
    return parser.parse_args()

def save_stsb_info(output_file, split="test"):
    sts_dataset_path = 'datasets/stsbenchmark.tsv.gz'
    batch_size = 96
    if split == "test":
        test_samples = []
        with gzip.open(sts_dataset_path, 'rt', encoding='utf8') as fIn:
            reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
            for row in reader:
                if row['split'] == 'test':
                    score = float(row['score']) / 5.0 #Normalize score to range 0 ... 1
                    test_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=score))
        print(f"Number of samples: {len(test_samples)}")
        evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, batch_size=batch_size, name='nouse')
    elif split == "dev":
        dev_samples = []
        with gzip.open(sts_dataset_path, 'rt', encoding='utf8') as fIn:
            reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
            for row in reader:
                if row['split'] == 'dev':
                    score = float(row['score']) / 5.0 #Normalize score to range 0 ... 1
                    dev_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=score))
        print(f"Number of samples: {len(dev_samples)}")
        evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, batch_size=batch_size, name='nouse')

    model = SentenceTransformer("./final_output/bert-base-uncased/")
    model[0].feature_cache = []
    evaluator(model, output_path="./final_output/bert-base-uncased/")
    print(f"Number of texts: {len(model[0].feature_cache)}")
    with open(output_file, "w") as f:
        for obj in model[0].feature_cache:
            new_lst = [str(item) for item in obj["input_id"] if item not in [0]]
            f.write(f"{' '.join(new_lst)}\n")

def load_sample_features(input_file):
    with open(input_file, "r") as f:
        lines = f.readlines()
    features = [[int(item) for item in line.strip().split()] for line in lines]
    return features

def compute_token_features(sample_features, method="freq"):
    assert method in ["freq", "tfidf"]
    if method == "freq":
        id2freq = {}
        for sample_feature in sample_features:
            for token_id in sample_feature:
                if token_id not in id2freq:
                    id2freq[token_id] = 1
                else:
                    id2freq[token_id] += 1
        return id2freq
    elif method == "tfidf":
        raise NotImplementedError
def filter_freq_rank_leq_than(num):
    def token_valid(token_id, token2feature, token2rank):
        if token_id == 0:
            return False
        if token_id not in token2rank:
            return True
        if token2rank[token_id] <= num:
            return False
        else:
            return True
    return token_valid

class TokenChecker:
    def __init__(self, check_func, token2feature, token2rank):
        self.check_func = check_func
        self.token2feature = token2feature
        self.token2rank = token2rank
    def __call__(self, token_id):
        return self.check_func(token_id, self.token2feature, self.token2rank)

def restrict_eval_nli_unsup(model_path, output_path, main_similarity=SimilarityFunction.COSINE, last2avg=True, restrict_method="freq", num_filter_freq_rank_leq_than=50):
    if not os.path.exists(output_path):
        os.mkdir(output_path)
                
    sample_features = load_sample_features("./tmp/stsb_test_features.txt")
    token2feature = compute_token_features(sample_features, method=restrict_method)
    token2feature_tuple = [(k, v) for k, v in token2feature.items()]
    sorted_tuple = sorted(token2feature_tuple, key=lambda item: item[1], reverse=True)
    with open(os.path.join(output_path, "token_features.txt"), "w") as f:
        for token_id, feature in sorted_tuple:
            f.write(f"{token_id}\t{feature}\n")
    token2rank = {token_id: idx for idx, (token_id, _) in enumerate(sorted_tuple)}
    json.dump(token2rank, open(os.path.join(output_path, "token_rank.json"), "w"), indent=4)
    
    model = load_model(model_path, last2avg=last2avg)
    model[1].token_checker = TokenChecker(filter_freq_rank_leq_than(num_filter_freq_rank_leq_than), token2feature, token2rank)
        
    score_sts12 = eval_sts12(model, output_path=output_path, main_similarity=main_similarity)
    score_sts13 = eval_sts13(model, output_path=output_path, main_similarity=main_similarity)
    score_sts14 = eval_sts14(model, output_path=output_path, main_similarity=main_similarity)
    score_sts15 = eval_sts15(model, output_path=output_path, main_similarity=main_similarity)
    score_sts16 = eval_sts16(model, output_path=output_path, main_similarity=main_similarity)
    score_stsb = eval_stsbenchmark(model, output_path=output_path, main_similarity=main_similarity)
    score_sickr = eval_sickr(model, output_path=output_path, main_similarity=main_similarity)
    score_sum = score_sts12 + score_sts13 + score_sts14 + score_sts15 + score_sts16 + score_stsb + score_sickr
    score_avg = score_sum / 7.0
    logging.info(f"Average score in unsupervised experiments: {score_avg:.6f}")
    json.dump({
        "sts12": score_sts12,
        "sts13": score_sts13,
        "sts14": score_sts14,
        "sts15": score_sts15,
        "sts16": score_sts16,
        "stsb": score_stsb,
        "sickr": score_sickr,
        "average": score_avg
    }, open(os.path.join(output_path, "summary.json"), "w"), indent=4)
    return score_avg


if __name__ == "__main__":
#     save_stsb_info("tmp/stsb_test_features.txt", split="test")
#     save_stsb_info("tmp/stsb_dev_features.txt", split="dev")
    args = parse_args()
    restrict_eval_nli_unsup(args.model_path, args.output_dir, restrict_method=args.filter_by, num_filter_freq_rank_leq_than=args.num_filter_freq_rank_leq_than)