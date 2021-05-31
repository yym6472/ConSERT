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

## Get Started

1. Download pre-trained language model (e.g. bert-base-uncased) from [HuggingFace's Library](https://huggingface.co/bert-base-uncased)
2. Download STS datasets to `./data` folder using [SentEval toolkit](https://github.com/facebookresearch/SentEval/blob/master/data/downstream/get_transfer_data.bash)
3. Run the following script to run the unsupervised experiment:
    ```bash
    python3 main.py --no_pair --seed 1 --use_apex_amp --apex_amp_opt_level O1 --batch_size 96 --max_seq_length 64 --evaluation_steps 200 --add_cl --cl_loss_only --cl_rate 0.15 --temperature 0.1 --learning_rate 0.0000005 --train_data stssick --num_epochs 10 --da_final_1 feature_cutoff --da_final_2 shuffle --cutoff_rate_final_1 0.2 --model_name_or_path [PRETRAINED_BERT_FOLDER] --model_save_path ./output/unsup-base-feature_cutoff-shuffle --force_del --no_dropout --patience 10
    ```
    where \[PRETRAINED_BERT_FOLDER\] should be replaced to the folder that contains downloaded pre-trained language model

## Citation
```
@article{yan2021consert,
  title={ConSERT: A Contrastive Framework for Self-Supervised Sentence Representation Transfer},
  author={Yan, Yuanmeng and Li, Rumei and Wang, Sirui and Zhang, Fuzheng and Wu, Wei and Xu, Weiran},
  journal={arXiv preprint arXiv:2105.11741},
  year={2021}
}
```