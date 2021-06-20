import os
import json
import random
import logging
import argparse
import io

from sentence_transformers import InputExample, LoggingHandler


logging.basicConfig(format='%(asctime)s - %(filename)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

def save_samples(samples, output_file):
    with open(output_file, "w", encoding="utf-8") as f_out:
        for sample in samples:
            line = "\t".join(sample.texts)
            f_out.write(f"{line}\n")

def load_paired_samples(input_file: str, label_file: str, need_label: bool = False, scale=5.0, no_pair=False):
    if need_label:
        assert not no_pair, "Only paired texts need label"
    with open(input_file, "r") as f:
        input_lines = [line.strip() for line in f.readlines()]
    label_lines = [0]*len(input_lines) # dummy
    if label_file!="":
        with open(label_file, "r") as f:
            label_lines = [line.strip() for line in f.readlines()]
    if need_label:
        new_input_lines, new_label_lines = [], []
        for idx in range(len(label_lines)):
            if label_lines[idx]:
                new_input_lines.append(input_lines[idx])
                new_label_lines.append(label_lines[idx])
        input_lines = new_input_lines
        label_lines = new_label_lines
    samples = []
    for input_line, label_line in zip(input_lines, label_lines):
        sentences = input_line.split("\t")
        if len(sentences)==2:
            sent1, sent2 = sentences
        else:
            sent1, sent2 = sentences[0], None
        if need_label:
            samples.append(InputExample(texts=[sent1, sent2], label=float(label_line)/scale))
        else:
            if no_pair:
                samples.append(InputExample(texts=[sent1]))
                if sent2:
                    samples.append(InputExample(texts=[sent2]))
            else:
                samples.append(InputExample(texts=[sent1, sent2]))
    return samples

def load_sts(year, dataset_names, need_label=False, no_pair=False):
    logging.info(f"Loading STS{year} dataset")
    sts_data_path = f"./data/downstream/STS/STS{year}-en-test"
    
    all_samples = []
    for dataset_name in dataset_names:
        input_file = os.path.join(sts_data_path, f"STS.input.{dataset_name}.txt")
        label_file = os.path.join(sts_data_path, f"STS.gs.{dataset_name}.txt")
        sub_samples = load_paired_samples(input_file, label_file, need_label=need_label, no_pair=no_pair)
        all_samples.extend(sub_samples)
    logging.info(f"Loaded examples from STS{year} dataset, total {len(all_samples)} examples")
    return all_samples

def load_senteval_binary(task_name, need_label=False, use_all_unsupervised_texts=True, no_pair=True):
    if task_name=="mr":
        dataset_names = ['rt-polarity.pos', 'rt-polarity.neg']
        data_path = f"./data/downstream/MR"
    elif task_name=="cr":
        dataset_names = ['custrev.pos', 'custrev.neg']
        data_path = f"./data/downstream/CR"
    elif task_name=="subj":
        dataset_names = ['subj.objective', 'subj.subjective']
        data_path = f"./data/downstream/SUBJ"
    elif task_name=="mpqa":
        dataset_names = ['mpqa.pos', 'mpqa.neg']
        data_path = f"./data/downstream/MPQA"
    all_samples = []
    for name in dataset_names:
        input_file = os.path.join(data_path, name)
        sub_samples = load_paired_samples(input_file, "", need_label=False, no_pair=True)
        all_samples.extend(sub_samples)
    logging.info(f"Loaded examples from {task_name.upper()} dataset, total {len(all_samples)} examples")
    return all_samples

def load_senteval_sst(need_label=False, use_all_unsupervised_texts=True, no_pair=True):
    data_path = f"./data/downstream/SST/binary"
    samples = []
    for name in ["sentiment-dev","sentiment-test","sentiment-train"]:
        input_file = os.path.join(data_path, name)
        for ln in open(input_file):
            sent = ln.strip().split("\t")[0]
            samples.append(InputExample(texts=[sent]))
    logging.info(f"Loaded examples from SST dataset, total {len(samples)} examples")
    return samples

def load_senteval_trec(need_label=False, use_all_unsupervised_texts=True, no_pair=True):
    data_path = f"./data/downstream/TREC"
    samples = []
    for name in ["train_5500.label","TREC_10.label"]:
        input_file = os.path.join(data_path, name)
        for ln in io.open(input_file, 'r', encoding='latin-1'):
            target, sample = ln.strip().split(':', 1)
            sample = sample.split(' ', 1)[1]
            samples.append(InputExample(texts=[sample]))
    logging.info(f"Loaded examples from TREC dataset, total {len(samples)} examples")
    return samples

def load_senteval_mrpc(need_label=False, use_all_unsupervised_texts=True, no_pair=True):
    data_path = f"./data/downstream/MRPC"
    samples = []
    for name in ["msr_paraphrase_test.txt","msr_paraphrase_train.txt"]:
        input_file = os.path.join(data_path, name)
        for ln in open(input_file):
            text = ln.strip().split('\t')
            samples.append(InputExample(texts=[text[3]]))
            samples.append(InputExample(texts=[text[4]]))
    logging.info(f"Loaded examples from MRPC dataset, total {len(samples)} examples")
    return samples


def load_sts12(need_label=False, use_all_unsupervised_texts=True, no_pair=False):
    dataset_names = ["MSRpar", "MSRvid", "SMTeuroparl", "surprise.OnWN", "surprise.SMTnews"]
    return load_sts("12", dataset_names, need_label=need_label, no_pair=no_pair)
    
def load_sts13(need_label=False, use_all_unsupervised_texts=True, no_pair=False):
    dataset_names = ["headlines", "OnWN", "FNWN"]
    return load_sts("13", dataset_names, need_label=need_label, no_pair=no_pair)

def load_sts14(need_label=False, use_all_unsupervised_texts=True, no_pair=False):
    dataset_names = ["images", "OnWN", "tweet-news", "deft-news", "deft-forum", "headlines"]
    return load_sts("14", dataset_names, need_label=need_label, no_pair=no_pair)

def load_sts15(need_label=False, use_all_unsupervised_texts=True, no_pair=False):
    dataset_names = ["answers-forums", "answers-students", "belief", "headlines", "images"]
    return load_sts("15", dataset_names, need_label=need_label, no_pair=no_pair)

def load_sts16(need_label=False, use_all_unsupervised_texts=True, no_pair=False):
    dataset_names = ["answer-answer", "headlines", "plagiarism", "postediting", "question-question"]
    return load_sts("16", dataset_names, need_label=need_label, no_pair=no_pair)

def load_stsbenchmark(need_label=False, use_all_unsupervised_texts=True, no_pair=False):
    if need_label:
        assert not no_pair, "Only paired texts need label"
    logging.info("Loading STSBenchmark dataset")
    all_samples = []
    if use_all_unsupervised_texts:
        splits = ["train", "dev", "test"]
    else:
        splits = ["test"]
    for split in splits:
        sts_benchmark_data_path = f"./data/downstream/STS/STSBenchmark/sts-{split}.csv"
        with open(sts_benchmark_data_path, "r") as f:
            lines = [line.strip() for line in f if line.strip()]
        samples = []
        for line in lines:
            _, _, _, _, label, sent1, sent2 = line.split("\t")
            if need_label:
                samples.append(InputExample(texts=[sent1, sent2], label=float(label) / 5.0))
            else:
                if no_pair:
                    samples.append(InputExample(texts=[sent1]))
                    samples.append(InputExample(texts=[sent2]))
                else:
                    samples.append(InputExample(texts=[sent1, sent2]))
        all_samples.extend(samples)
    logging.info(f"Loaded examples from STSBenchmark dataset, total {len(all_samples)} examples")
    return all_samples

def load_sickr(need_label=False, use_all_unsupervised_texts=True, no_pair=False):
    if need_label:
        assert not no_pair, "Only paired texts need label"
    logging.info("Loading SICK (relatedness) dataset")
    all_samples = []
    if use_all_unsupervised_texts:
        splits = ["train", "trial", "test_annotated"]
    else:
        splits = ["test_annotated"]
    for split in splits:
        sick_data_path = f"./data/downstream/SICK/SICK_{split}.txt"
        with open(sick_data_path, "r") as f:
            lines = [line.strip() for line in f if line.strip()]
        samples = []
        for line in lines[1:]:
            _, sent1, sent2, label, _ = line.split("\t")
            if need_label:
                samples.append(InputExample(texts=[sent1, sent2], label=float(label) / 5.0))
            else:
                if no_pair:
                    samples.append(InputExample(texts=[sent1]))
                    samples.append(InputExample(texts=[sent2]))
                else:
                    samples.append(InputExample(texts=[sent1, sent2]))
        all_samples.extend(samples)
    logging.info(f"Loaded examples from SICK dataset, total {len(all_samples)} examples")
    return all_samples

def load_datasets(datasets=None, need_label=False, use_all_unsupervised_texts=True, no_pair=False):
    load_function_mapping = {
        "sts12": load_sts12,
        "sts13": load_sts13,
        "sts14": load_sts14,
        "sts15": load_sts15,
        "sts16": load_sts16,
        "stsb": load_stsbenchmark,
        "sickr": load_sickr
    }
    datasets = datasets or ["sts12", "sts13", "sts14", "sts15", "sts16", "stsb", "sickr"]
    all_samples = []
    for dataset in datasets:
        func = load_function_mapping[dataset]
        all_samples.extend(func(need_label=need_label, use_all_unsupervised_texts=use_all_unsupervised_texts, no_pair=no_pair))
    logging.info(f"Loaded data from datasets {datasets}, total number of samples {len(all_samples)}")
    return all_samples

def load_chinese_tsv_data(dataset_name, split, max_num_samples=None):
    assert dataset_name in ("atec_ccks", "bq", "lcqmc", "pawsx", "stsb")
    assert split in ("train", "dev", "test")
    base_data_path = "./data/chinese"
    data_file = os.path.join(base_data_path, dataset_name, f"{split}.tsv")
    all_samples = []
    with open(data_file) as f:
        lines = f.readlines()
    for line in lines:
        sent1, sent2, label = line.strip().split("\t")
        if split == "train":
            all_samples.append(InputExample(texts=[sent1]))
            all_samples.append(InputExample(texts=[sent2]))
        else:
            all_samples.append(InputExample(texts=[sent1, sent2], label=float(label)))
    if max_num_samples is not None and max_num_samples < len(all_samples):
        all_samples = random.sample(all_samples, max_num_samples)
    return all_samples


if __name__ == "__main__":
    samples = load_datasets(need_label=False, use_all_unsupervised_texts=True, no_pair=True)
    print(samples[0])