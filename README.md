# ConSERT

Code for our ACL 2021 paper - [ConSERT: A Contrastive Framework for Self-Supervised Sentence Representation Transfer](https://arxiv.org/abs/2105.11741)

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

1. Download pre-trained language model (e.g. bert-base-uncased) to folder `./bert-base-uncased` from [HuggingFace's Library](https://huggingface.co/bert-base-uncased)
2. Download STS datasets to `./data` folder by running `cd data && bash get_transfer_data.bash`. The script is modified from [SentEval toolkit](https://github.com/facebookresearch/SentEval/blob/master/data/downstream/get_transfer_data.bash)
3. Run the scripts in the folder `./scripts` to reproduce our experiments. For example, run the following script to train unsupervised consert-base:
    ```bash
    bash scripts/unsup-consert-base.sh
    ```

## Pre-trained Models & Results

### English STS Tasks

| ID | Model                                                         | STS12 | STS13 | STS14 | STS15 | STS16 | STSb | SICK-R | Avg. |
|----|---------------------------------------------------------------|:-----:|:-----:|:-----:|:-----:|:-----:|:----:|:------:|:----:|
| 1  | unsup-consert-base [\[Google Drive\]](https://drive.google.com/file/d/1KIbrhhIfhxO_4b0tbpdhLGkrjr_bhlaa/view?usp=sharing) [\[百度云q571\]](https://pan.baidu.com/s/1Nh_ypA-kP2cXIt3DE_0aXg)            |   64.64    |  78.49     |  69.07     | 79.72      |  75.95     |  73.97    |   67.31     |  72.74    |
| 2  | unsup-consert-large [\[Google Drive\]](https://drive.google.com/file/d/1sYlJ_O43QHC1WYY9h3HW_bWLhxvb7lEH/view?usp=sharing) [\[百度云9fm1\]](https://pan.baidu.com/s/1XBonWQhDp68-TUfVV8NBBQ)           |  70.28     |   83.23    |  73.80     |  82.73     |    77.14   |  77.74    |  70.19      |  76.45    |
| 3  | sup-sbert-base (re-impl.) [\[Google Drive\]]() [\[百度云\]]()     |       |       |       |       |       |      |        |      |
| 4  | sup-sbert-large (re-impl.) [\[Google Drive\]]() [\[百度云\]]()    |       |       |       |       |       |      |        |      |
| 5  | sup-consert-joint-base [\[Google Drive\]]() [\[百度云\]]()        |       |       |       |       |       |      |        |      |
| 6  | sup-consert-joint-large [\[Google Drive\]]() [\[百度云\]]()       |       |       |       |       |       |      |        |      |
| 7  | sup-consert-sup-unsup-base [\[Google Drive\]]() [\[百度云\]]()    |       |       |       |       |       |      |        |      |
| 8  | sup-consert-sup-unsup-large [\[Google Drive\]]() [\[百度云\]]()   |       |       |       |       |       |      |        |      |
| 9  | sup-consert-joint-unsup-base [\[Google Drive\]]() [\[百度云\]]()  |       |       |       |       |       |      |        |      |
| 10 | sup-consert-joint-unsup-large [\[Google Drive\]]() [\[百度云\]]() |       |       |       |       |       |      |        |      |

Note:
1. All the *base* models are trained from `bert-base-uncased` and the *large* models are trained from `bert-large-uncased`.
2. For the unsupervised transfer, we merge all unlabeled texts from 7 STS datasets (STS12-16, STSbenchmark and SICK-Relatedness) as the training data (total 89192 sentences), and use the STSbenchmark dev split (including 1500 human-annotated sentence pairs) to select the best checkpoint.
3. The sentence representations are obtained by averaging the token embeddings at the last two layers of BERT.
4. For model 2 to 10, we re-trained them on a single GeForce RTX 3090 with pytorch 1.8.1 and cuda 11.1 (rather than V100, pytorch 1.6.0 and cuda 10.0 in our initial experiments) and changed the `max_seq_length` from 64 to 40 to reduce the required GPU memory (only for *large* models). Consequently, the results shown here may be slightly different from those reported in our paper.

### Chinese STS Tasks

To be added.

## Citation
```
@article{yan2021consert,
  title={ConSERT: A Contrastive Framework for Self-Supervised Sentence Representation Transfer},
  author={Yan, Yuanmeng and Li, Rumei and Wang, Sirui and Zhang, Fuzheng and Wu, Wei and Xu, Weiran},
  journal={arXiv preprint arXiv:2105.11741},
  year={2021}
}
```
