# ConSERT

Code for our ACL 2021 paper - [ConSERT: A Contrastive Framework for Self-Supervised Sentence Representation Transfer](https://aclanthology.org/2021.acl-long.393/)

## Requirements

```
torch==1.6.0
cudatoolkit==10.0.103
cudnn==7.6.5
sentence-transformers==0.3.9
transformers==3.4.0
tensorboardX==2.1
pandas==1.1.5
sentencepiece==0.1.85
matplotlib==3.4.1
apex==0.1.0
```

To install apex, run:
```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir ./
```

## Get Started

### English STS Tasks

1. Download pre-trained language model (e.g. bert-base-uncased) to folder `./bert-base-uncased` from [HuggingFace's Library](https://huggingface.co/bert-base-uncased)
2. Download STS datasets to `./data` folder by running `cd data && bash get_transfer_data.bash`. The script is modified from [SentEval toolkit](https://github.com/facebookresearch/SentEval/blob/master/data/downstream/get_transfer_data.bash)
3. Run the scripts in the folder `./scripts` to reproduce our experiments. For example, run the following script to train unsupervised consert-base:
    ```bash
    bash scripts/unsup-consert-base.sh
    ```

*Update*: Recently we update the support for RoBERTa model. Download the RoBERTa model to folder e.g. `./roberta-base` and use scripts in `./scripts/roberta` instead.

### Chinese STS Tasks

1. Download the pre-trained language model (e.g. chinese-roberta-wwm-ext) to folder `./chinese-roberta-wwm-ext` from [HuggingFace's Library](https://huggingface.co/hfl/chinese-roberta-wwm-ext)
2. Download the chinese STS datasets to `./data` folder by running `cd data && bash get_chinese_sts_data.bash`
3. Run the scripts in the folder `./scripts/chinese` to train models for chinese STS tasks. For example, run the following script to train *base* model for `atec_ccks` task:
    ```bash
    bash scripts/chinese/unsup-consert-base-atec_ccks.sh
    ```
   The `--chinese_dataset` option in `main.py` is used to select which chinese STS dataset to use.

## Pre-trained Models & Results

### English STS Tasks

| ID | Model                                                         | STS12 | STS13 | STS14 | STS15 | STS16 | STSb | SICK-R | Avg. |
|----|---------------------------------------------------------------|:-----:|:-----:|:-----:|:-----:|:-----:|:----:|:------:|:----:|
| -  | bert-base-uncased (baseline) | 35.20 | 59.53 | 49.37 | 63.39 | 62.73 | 48.18 | 58.60 | 53.86 |
| -  | bert-large-uncased (baseline) | 33.06 | 57.64 | 47.95 | 55.83 | 62.42 | 49.66 | 53.87 | 51.49 |
| 1  | unsup-consert-base [\[Google Drive\]](https://drive.google.com/file/d/1KIbrhhIfhxO_4b0tbpdhLGkrjr_bhlaa/view?usp=sharing) [\[百度云q571\]](https://pan.baidu.com/s/1Nh_ypA-kP2cXIt3DE_0aXg)            |   64.64    |  78.49     |  69.07     | 79.72      |  75.95     |  73.97    |   67.31     |  72.74    |
| 2  | unsup-consert-large [\[Google Drive\]](https://drive.google.com/file/d/1sYlJ_O43QHC1WYY9h3HW_bWLhxvb7lEH/view?usp=sharing) [\[百度云9fm1\]](https://pan.baidu.com/s/1XBonWQhDp68-TUfVV8NBBQ)           |  70.28     |   83.23    |  73.80     |  82.73     |    77.14   |  77.74    |  70.19      |  76.45    |
| 3  | sup-sbert-base (re-impl.) [\[Google Drive\]](https://drive.google.com/file/d/1Q5wN6FRikBKEJ3jeJlPGPQ0WWEf0mkdi/view?usp=sharing) [\[百度云msqy\]](https://pan.baidu.com/s/1IDztb2LEdje4_aIhQHOhoQ)     |   69.93    |   76.00    |  72.15     |   78.59    | 73.53      |   76.10   |    73.01    |  74.19    |
| 4  | sup-sbert-large (re-impl.) [\[Google Drive\]](https://drive.google.com/file/d/1JJjvxiyJEdS62GpH0mDz1qK0AApBPQ3Z/view?usp=sharing) [\[百度云0oir\]](https://pan.baidu.com/s/1yYlG2Us7CBVEt_kLIaazrw)    |   73.06    |  77.77     |   75.21    |   81.63    |   77.30    |   79.74   |   74.75     |   77.07   |
| 5  | sup-consert-joint-base [\[Google Drive\]](https://drive.google.com/file/d/1uCNWUWPyjuTfyIqP4zsVNNxeGRNyXOrw/view?usp=sharing) [\[百度云jks5\]](https://pan.baidu.com/s/1R3QIA99_RTY5CeuJa8Q6UA)        |  70.92     |  79.98     |   74.88    |   81.76    |   76.46    |    78.99  |   78.15     |  77.31    |
| 6  | sup-consert-joint-large [\[Google Drive\]](https://drive.google.com/file/d/1cE8wauMEHGKn52kyCpOcIKKeJ7p3-cTq/view?usp=sharing) [\[百度云xua4\]](https://pan.baidu.com/s/1nNrwJNoctWYRYaUjrz5qRw)       |  73.15     |  81.45     |   77.04    |   83.32    |  77.28     |    81.15  |   78.34     |  78.82    |
| 7  | sup-consert-sup-unsup-base [\[Google Drive\]](https://drive.google.com/file/d/1w3SmCC6ibm8NSvh80ZB0ERDonpPvm8mz/view?usp=sharing) [\[百度云5mc8\]](https://pan.baidu.com/s/1Kgi14KXeby0eFCzraCYfDg)    |   73.02    |  84.86     |  77.32     |   82.70    |  78.20     |    81.34  |   75.00     |  78.92    |
| 8  | sup-consert-sup-unsup-large [\[Google Drive\]](https://drive.google.com/file/d/15E8nAFprUFTv5KPPgxvedgigXXwcRVeT/view?usp=sharing) [\[百度云tta1\]](https://pan.baidu.com/s/1UeLwlUUWR3QogqTeUQZ0iw)   |  74.99     |   85.58    | 79.17      |   84.25    |   80.19    |    83.17  |   77.43     |  80.68    |
| 9  | sup-consert-joint-unsup-base [\[Google Drive\]](https://drive.google.com/file/d/14jcb8NCDB3PGr0LZI_tZcmBATxkB4OEc/view?usp=sharing) [\[百度云cf07\]](https://pan.baidu.com/s/15fbohm8TdZLHhW9NEonl_Q)  |  74.46     |   84.19    |  77.08     |   83.77    |  78.55     |     81.37 |    77.01    |  79.49    |
| 10 | sup-consert-joint-unsup-large [\[Google Drive\]](https://drive.google.com/file/d/1xo5QrlG_TJ6NetqSX1nYtaVLPzee2kpx/view?usp=sharing) [\[百度云v5x5\]](https://pan.baidu.com/s/1b53Tk0ZwTvlti_OSfHDhzg) |  76.93     |  85.20     |  78.69     |   85.44    |   79.34    |    82.93  |  76.71      |  80.75    |

Note:
1. All the *base* models are trained from `bert-base-uncased` and the *large* models are trained from `bert-large-uncased`.
2. For the unsupervised transfer, we merge all unlabeled texts from 7 STS datasets (STS12-16, STSbenchmark and SICK-Relatedness) as the training data (total 89192 sentences), and use the STSbenchmark dev split (including 1500 human-annotated sentence pairs) to select the best checkpoint.
3. The sentence representations are obtained by averaging the token embeddings at the last two layers of BERT.
4. For model 2 to 10, we re-trained them on a single GeForce RTX 3090 with pytorch 1.8.1 and cuda 11.1 (rather than V100, pytorch 1.6.0 and cuda 10.0 in our initial experiments) and changed the `max_seq_length` from 64 to 40 to reduce the required GPU memory (only for *large* models). Consequently, the results shown here may be slightly different from those reported in our paper.

### Chinese STS Tasks

| ID | Model                              | atec_ccks | bq | lcqmc | pawsx | stsb |
|----|------------------------------------|:-----:|:-----:|:-----:|:-----:|:-----:|
| -  | chinese-roberta-wwm-ext (baseline) | 11.28 | 40.21 | 59.89 | 09.35 | 60.86 |
| -  | chinese-roberta-wwm-ext-large (baseline) | 13.75 | 36.77 | 60.36 | 09.94 | 58.72 |
| 1  | unsup-consert-base-atec_ccks  | **27.39** | 47.61 | 60.70  | 08.17 | 64.74 |
| 2  | unsup-consert-base-bq         | 11.55 | **47.20** | 61.47  | 08.47 | 64.44 |
| 3  | unsup-consert-base-lcqmc      | 04.57 | 38.15 | **67.34**  | 08.80 | 67.78 |
| 4  | unsup-consert-base-pawsx      | 08.66 | 38.35 | 61.97  | **09.36** | 63.89 |
| 5  | unsup-consert-base-stsb       | 07.29 | 40.81 | 67.39  | 06.24  | **72.82** |
| 6  | unsup-consert-large-atec_ccks | **29.92** | 47.50 | 59.83 | 09.72 | 66.37 |
| 7  | unsup-consert-large-bq        | 15.47 | **47.08** | 60.10 | 10.04 | 66.94 |
| 8  | unsup-consert-large-lcqmc     | 11.40 | 37.79 | **66.45** | 10.81 | 67.68 |
| 9  | unsup-consert-large-pawsx     | 11.25 | 36.25 | 65.15 | **11.38** | 67.45 |
| 10 | unsup-consert-large-stsb      | 08.00 | 40.85 | 67.72 | 10.13 | **74.50** |

Note:
1. All the *base* models are trained from `hfl/chinese-roberta-wwm-ext` and the *large* models are trained from `hfl/chinese-roberta-wwm-ext-large`. 
2. For each model, we train it on *single* chinese STS dataset but evaluate it on all 5 datasets. **The bold numbers** indicate that the model is evaluated on the same dataset it trained on.
3. The sentence representations are also obtained by averaging the token embeddings at the last two layers of BERT.
4. For *base* models, we set `batch_size` to 96 and `max_seq_length` to 64, while for *large* models, we set `batch_size` to 32 and `max_seq_length` to 40 to reduce the required GPU memory.

## Citation
```
@inproceedings{yan-etal-2021-consert,
    title = "{C}on{SERT}: A Contrastive Framework for Self-Supervised Sentence Representation Transfer",
    author = "Yan, Yuanmeng  and
      Li, Rumei  and
      Wang, Sirui  and
      Zhang, Fuzheng  and
      Wu, Wei  and
      Xu, Weiran",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.acl-long.393",
    doi = "10.18653/v1/2021.acl-long.393",
    pages = "5065--5075",
    abstract = "Learning high-quality sentence representations benefits a wide range of natural language processing tasks. Though BERT-based pre-trained language models achieve high performance on many downstream tasks, the native derived sentence representations are proved to be collapsed and thus produce a poor performance on the semantic textual similarity (STS) tasks. In this paper, we present ConSERT, a Contrastive Framework for Self-Supervised SEntence Representation Transfer, that adopts contrastive learning to fine-tune BERT in an unsupervised and effective way. By making use of unlabeled texts, ConSERT solves the collapse issue of BERT-derived sentence representations and make them more applicable for downstream tasks. Experiments on STS datasets demonstrate that ConSERT achieves an 8{\%} relative improvement over the previous state-of-the-art, even comparable to the supervised SBERT-NLI. And when further incorporating NLI supervision, we achieve new state-of-the-art performance on STS tasks. Moreover, ConSERT obtains comparable results with only 1000 samples available, showing its robustness in data scarcity scenarios.",
}

```
