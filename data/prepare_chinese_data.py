import os
import sys
import pandas


def prepare_atec_ccks(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    input_filenames = {
        "train": "train.csv",
        "dev": "dev.csv",
        "test": "test.csv"
    }
    for split in ("train", "dev", "test"):
        input_file = os.path.join(input_dir, input_filenames[split])
        with open(input_file, "r", encoding="utf-8") as f:
            lines = f.readlines()[1:]
        with open(os.path.join(output_dir, f"{split}.tsv"), "w", encoding="utf-8") as f:
            for line in lines:
                sent1, sent2, label = line.strip().split("\t")
                sent1, sent2, label = sent1.strip(), sent2.strip(), label.strip()
                assert all(item and "\t" not in item and "\n" not in item for item in (sent1, sent2, label))
                f.write(f"{sent1}\t{sent2}\t{label}\n")

def prepare_bq(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    input_filenames = {
        "train": "train.csv",
        "dev": "dev.csv",
        "test": "test.csv"
    }
    for split in ("train", "dev", "test"):
        input_file = os.path.join(input_dir, input_filenames[split])
        table = pandas.read_csv(input_file)
        with open(os.path.join(output_dir, f"{split}.tsv"), "w", encoding="utf-8") as f:
            for idx in range(len(table)):
                sent1, sent2, label = table["sentence1"][idx], table["sentence2"][idx], str(table["label"][idx])
                sent1, sent2, label = sent1.strip(), sent2.strip(), label.strip()
                assert all(item and "\t" not in item and "\n" not in item for item in (sent1, sent2, label))
                f.write(f"{sent1}\t{sent2}\t{label}\n")

def prepare_lcqmc(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    input_filenames = {
        "train": "train.txt",
        "dev": "dev.txt",
        "test": "test.txt"
    }
    for split in ("train", "dev", "test"):
        input_file = os.path.join(input_dir, input_filenames[split])
        with open(input_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        with open(os.path.join(output_dir, f"{split}.tsv"), "w", encoding="utf-8") as f:
            for line in lines:
                sent1, sent2, label = line.strip().split("\t")
                sent1, sent2, label = sent1.strip(), sent2.strip(), label.strip()
                assert all(item and "\t" not in item and "\n" not in item for item in (sent1, sent2, label))
                f.write(f"{sent1}\t{sent2}\t{label}\n")

def prepare_pawsx(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    input_filenames = {
        "train": "translated_train.tsv",
        "dev": "dev_2k.tsv",
        "test": "test_2k.tsv"
    }
    for split in ("train", "dev", "test"):
        input_file = os.path.join(input_dir, input_filenames[split])
        with open(input_file, "r", encoding="utf-8") as f:
            lines = f.readlines()[1:]
        with open(os.path.join(output_dir, f"{split}.tsv"), "w", encoding="utf-8") as f:
            for line in lines:
                _, sent1, sent2, label = line.strip().split("\t")
                sent1, sent2, label = sent1.strip(), sent2.strip(), label.strip()
                if not all(item and "\t" not in item and "\n" not in item for item in (sent1, sent2, label)):
                    print(f"Ignored invalid line: {line}")
                    continue
                f.write(f"{sent1}\t{sent2}\t{label}\n")

def prepare_stsb(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    input_filenames = {
        "train": "cnsd-sts-train.txt",
        "dev": "cnsd-sts-dev.txt",
        "test": "cnsd-sts-test.txt"
    }
    for split in ("train", "dev", "test"):
        input_file = os.path.join(input_dir, input_filenames[split])
        with open(input_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        with open(os.path.join(output_dir, f"{split}.tsv"), "w", encoding="utf-8") as f:
            for line in lines:
                _, sent1, sent2, label = line.strip().split("||")
                sent1, sent2, label = sent1.strip(), sent2.strip(), label.strip()
                assert all(item and "\t" not in item and "\n" not in item for item in (sent1, sent2, label))
                f.write(f"{sent1}\t{sent2}\t{label}\n")


def main():
    data_path = sys.argv[1]
    os.mkdir(data_path)

    prepare_atec_ccks(
        input_dir="./NLP_Datasets/ATEC_CCKS/",
        output_dir=os.path.join(data_path, "atec_ccks")
    )
    prepare_bq(
        input_dir="./NLP_Datasets/BQ_corpus/",
        output_dir=os.path.join(data_path, "bq")
    )
    prepare_lcqmc(
        input_dir="./NLP_Datasets/LCQMC/",
        output_dir=os.path.join(data_path, "lcqmc")
    )
    prepare_pawsx(
        input_dir="./x-final/zh/",
        output_dir=os.path.join(data_path, "pawsx")
    )
    prepare_stsb(
        input_dir="./STS-B/",
        output_dir=os.path.join(data_path, "stsb")
    )


if __name__ == "__main__":
    main()