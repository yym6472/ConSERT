import os
import json
import logging
import sys

from sentence_transformers import SentenceTransformer, InputExample, LoggingHandler
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, SimilarityFunction


logging.basicConfig(format='%(asctime)s - %(filename)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])


def load_model(model_path: str):
    model = SentenceTransformer(model_path)
    logging.info("Model successfully loaded")
    return model

def load_paired_samples(input_file: str, label_file: str, scale=5.0):
    with open(input_file, "r") as f:
        input_lines = [line.strip() for line in f.readlines()]
    with open(label_file, "r") as f:
        label_lines = [line.strip() for line in f.readlines()]
    new_input_lines, new_label_lines = [], []
    for idx in range(len(label_lines)):
        if label_lines[idx]:
            new_input_lines.append(input_lines[idx])
            new_label_lines.append(label_lines[idx])
    input_lines = new_input_lines
    label_lines = new_label_lines
    samples = []
    for input_line, label_line in zip(input_lines, label_lines):
        sent1, sent2 = input_line.split("\t")
        samples.append(InputExample(texts=[sent1, sent2], label=float(label_line)/scale))
    return samples

def eval_sts(model, year, dataset_names, batch_size=16, output_path="./", main_similarity=None):
    logging.info(f"Evaluation on STS{year} dataset")
    sts_data_path = f"./data/downstream/STS/STS{year}-en-test"
    
    all_samples = []
    results = {}
    sum_score = 0.0
    weighted_sum_score = 0.0
    for dataset_name in dataset_names:
        input_file = os.path.join(sts_data_path, f"STS.input.{dataset_name}.txt")
        label_file = os.path.join(sts_data_path, f"STS.gs.{dataset_name}.txt")
        sub_samples = load_paired_samples(input_file, label_file)
#         sub_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(sub_samples, batch_size=batch_size, name=f"sts-{year}-{dataset_name}", main_similarity=main_similarity)
#         sub_best_result = sub_evaluator(model, output_path=output_path)
#         results[dataset_name] = {
#             "num_samples": len(sub_samples),
#             "best_spearman": sub_best_result
#         }
#         sum_score += sub_best_result
#         weighted_sum_score += sub_best_result * len(sub_samples)
        all_samples.extend(sub_samples)
    logging.info(f"Loaded examples from STS{year} dataset, total {len(all_samples)} examples")
    
    evaluator = EmbeddingSimilarityEvaluator.from_input_examples(all_samples, batch_size=batch_size, name=f"sts-{year}", main_similarity=main_similarity)
    best_result = evaluator(model, output_path=output_path)
    logging.info(f"Results on STS{year}: {best_result:.6f}")
    results["all"] = {
        "num_samples": len(all_samples),
        "best_spearman_joint": best_result,
        "best_spearman_mean": sum_score / len(dataset_names),
        "best_spearman_wmean": weighted_sum_score / len(all_samples)
    }
    with open(os.path.join(output_path, f"STS{year}-results.json"), "w") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    return best_result

def eval_sts12(model, batch_size=16, output_path="./", main_similarity=None):
    dataset_names = ["MSRpar", "MSRvid", "SMTeuroparl", "surprise.OnWN", "surprise.SMTnews"]
    return eval_sts(model, "12", dataset_names, batch_size=batch_size, output_path=output_path, main_similarity=main_similarity)
    
def eval_sts13(model, batch_size=16, output_path="./", main_similarity=None):
    dataset_names = ["headlines", "OnWN", "FNWN"]
    return eval_sts(model, "13", dataset_names, batch_size=batch_size, output_path=output_path, main_similarity=main_similarity)

def eval_sts14(model, batch_size=16, output_path="./", main_similarity=None):
    dataset_names = ["images", "OnWN", "tweet-news", "deft-news", "deft-forum", "headlines"]
    return eval_sts(model, "14", dataset_names, batch_size=batch_size, output_path=output_path, main_similarity=main_similarity)

def eval_sts15(model, batch_size=16, output_path="./", main_similarity=None):
    dataset_names = ["answers-forums", "answers-students", "belief", "headlines", "images"]
    return eval_sts(model, "15", dataset_names, batch_size=batch_size, output_path=output_path, main_similarity=main_similarity)

def eval_sts16(model, batch_size=16, output_path="./", main_similarity=None):
    dataset_names = ["answer-answer", "headlines", "plagiarism", "postediting", "question-question"]
    return eval_sts(model, "16", dataset_names, batch_size=batch_size, output_path=output_path, main_similarity=main_similarity)

def eval_stsbenchmark(model, batch_size=16, output_path="./", main_similarity=None):
    logging.info("Evaluation on STSBenchmark dataset")
    sts_benchmark_data_path = "./data/downstream/STS/STSBenchmark/sts-test.csv"
    with open(sts_benchmark_data_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    samples = []
    for line in lines:
        _, _, _, _, label, sent1, sent2 = line.split("\t")
        samples.append(InputExample(texts=[sent1, sent2], label=float(label) / 5.0))
    logging.info(f"Loaded examples from STSBenchmark dataset, total {len(samples)} examples")
    
    evaluator = EmbeddingSimilarityEvaluator.from_input_examples(samples, batch_size=batch_size, name="sts-benchmark", main_similarity=main_similarity)
    best_result = evaluator(model, output_path=output_path)
    logging.info(f"Results on STSBenchmark: {best_result:.6f}")
    results = {
        "num_samples": len(samples),
        "best_spearman": best_result
    }
    with open(os.path.join(output_path, "STSBenchmark-results.json"), "w") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    return best_result

def eval_sickr(model, batch_size=16, output_path="./", main_similarity=None):
    logging.info("Evaluation on SICK (relatedness) dataset")
    sick_data_path = "./data/downstream/SICK/SICK_test_annotated.txt"
    with open(sick_data_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    samples = []
    for line in lines[1:]:
        _, sent1, sent2, label, _ = line.split("\t")
        samples.append(InputExample(texts=[sent1, sent2], label=float(label) / 5.0))
    logging.info(f"Loaded examples from SICK dataset, total {len(samples)} examples")
    
    evaluator = EmbeddingSimilarityEvaluator.from_input_examples(samples, batch_size=batch_size, name="sick-r", main_similarity=main_similarity)
    best_result = evaluator(model, output_path=output_path)
    logging.info(f"Results on SICK (relatedness): {best_result:.6f}")
    results = {
        "num_samples": len(samples),
        "best_spearman": best_result
    }
    with open(os.path.join(output_path, "SICK-R-results.json"), "w") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    return best_result

    
if __name__ == "__main__":
    model_path = sys.argv[1]
    main_similarity = SimilarityFunction.COSINE
    
    model = load_model(model_path)
    output_path = os.path.join(model_path, "sts_eval")
    if not os.path.exists(output_path):
        os.mkdir(output_path)
        
    logging.info(model_path)
    score_sum = 0.0
    score_sum += eval_stsbenchmark(model, output_path=output_path, main_similarity=main_similarity)
    score_sum += eval_sickr(model, output_path=output_path, main_similarity=main_similarity)
    score_sum += eval_sts12(model, output_path=output_path, main_similarity=main_similarity)
    score_sum += eval_sts13(model, output_path=output_path, main_similarity=main_similarity)
    score_sum += eval_sts14(model, output_path=output_path, main_similarity=main_similarity)
    score_sum += eval_sts15(model, output_path=output_path, main_similarity=main_similarity)
    score_sum += eval_sts16(model, output_path=output_path, main_similarity=main_similarity)
    logging.info(f"Average score in unsupervised experiments: {score_sum / 7:.6f}")