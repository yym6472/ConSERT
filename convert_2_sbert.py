"""
The system trains BERT (or any other transformer model like RoBERTa, DistilBERT etc.) on the SNLI + MultiNLI (AllNLI) dataset
with softmax loss function. At every 1000 training steps, the model is evaluated on the
STS benchmark dataset

Usage:
python training_nli.py --seed 1234

OR
python training_nli.py --seed 1234 --model_name_or_path bert-base-uncased
"""
from torch.utils.data import DataLoader
import math
from sentence_transformers import models, losses
from sentence_transformers import SentencesDataset, LoggingHandler, SentenceTransformer, util, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
import logging
from datetime import datetime
import sys
import os
import json
import gzip
import csv
import random
import torch
import numpy as np
import argparse


#### Set arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=40, help="Random seed for reproducing experimental results")
    parser.add_argument("--model_name_or_path", type=str, default="bert-base-uncased", help="The model path or model name of pre-trained model")
    parser.add_argument("--model_save_path", type=str, required=True, help="The model path or model name of converted model")
    parser.add_argument("--pooling_strategy", type=str, default="cls", help="BERT pooling strategy for output")
    return parser.parse_args()
args = parse_args()

print(f"Args\n{args}")

#### Added script to set random seed
def set_seed(seed: int, for_multi_gpu: bool):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if for_multi_gpu:
        torch.cuda.manual_seed_all(seed)
set_seed(args.seed, for_multi_gpu=False)

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

# #Check if dataset exsist. If not, download and extract  it
# nli_dataset_path = 'datasets/AllNLI.tsv.gz'
# sts_dataset_path = 'datasets/stsbenchmark.tsv.gz'

# if not os.path.exists(nli_dataset_path):
#     util.http_get('https://sbert.net/datasets/AllNLI.tsv.gz', nli_dataset_path)

# if not os.path.exists(sts_dataset_path):
#     util.http_get('https://sbert.net/datasets/stsbenchmark.tsv.gz', sts_dataset_path)

# # Read the dataset
# train_batch_size = args.batch_size


model_save_path = args.model_save_path
if not os.path.exists(model_save_path):
    os.mkdir(model_save_path)
with open(os.path.join(model_save_path, "seed.txt"), "w") as f:
    f.write(str(args.seed))
with open(os.path.join(model_save_path, "args.json"), "w") as f:
    json.dump(args.__dict__, f, indent=4, ensure_ascii=False)


# Use Huggingface/transformers model (like BERT, RoBERTa, XLNet, XLM-R) for mapping tokens to embeddings
word_embedding_model = models.Transformer(args.model_name_or_path)

# Apply pooling to get one fixed sized sentence vector
mean_pooling, cls_pooling, max_pooling, last2_pooling  = False, False, False, False
if args.pooling_strategy.lower()=="mean":
    mean_pooling = True
elif args.pooling_strategy.lower()=="max":
    max_pooling = True
elif args.pooling_strategy.lower()=="cls":
    cls_pooling = True
elif args.pooling_strategy.lower()=="last2":
    last2_pooling = True
else:
    raise NotImplementedError()

pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=mean_pooling,
                               pooling_mode_cls_token=cls_pooling,
                               pooling_mode_max_tokens=max_pooling,
                               pooling_mode_mean_last_2_tokens=last2_pooling)

model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

model.save(model_save_path)
print("model saved to", model_save_path)

# Read the AllNLI.tsv.gz file and create the training dataset
# logging.info("Read AllNLI train dataset")

# label2int = {"contradiction": 0, "entailment": 1, "neutral": 2}
# train_samples = []
# with gzip.open(nli_dataset_path, 'rt', encoding='utf8') as fIn:
#     reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
#     for row in reader:
#         if row['split'] == 'train':
#             label_id = label2int[row['label']]
#             train_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=label_id))


# train_dataset = SentencesDataset(train_samples, model=model)
# train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
# train_loss = losses.SoftmaxLoss(model=model, sentence_embedding_dimension=model.get_sentence_embedding_dimension(), num_labels=len(label2int))


# #Read STSbenchmark dataset and use it as development set
# logging.info("Read STSbenchmark dev dataset")
# dev_samples = []
# with gzip.open(sts_dataset_path, 'rt', encoding='utf8') as fIn:
#     reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
#     for row in reader:
#         if row['split'] == 'dev':
#             score = float(row['score']) / 5.0 #Normalize score to range 0 ... 1
#             dev_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=score))

# dev_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, batch_size=train_batch_size, name='sts-dev')

# # Configure the training
# num_epochs = args.num_epochs

# warmup_steps = math.ceil(len(train_dataset) * num_epochs / train_batch_size * 0.1) #10% of train data for warm-up
# logging.info("Warmup-steps: {}".format(warmup_steps))



# # Train the model
# model.fit(train_objectives=[(train_dataloader, train_loss)],
#           evaluator=dev_evaluator,
#           epochs=num_epochs,
#           evaluation_steps=1000,
#           warmup_steps=warmup_steps,
#           output_path=model_save_path,
#           use_apex_amp=args.use_apex_amp,
#           apex_amp_opt_level = args.apex_amp_opt_level
#           )



# ##############################################################################
# #
# # Load the stored model and evaluate its performance on STS benchmark dataset
# #
# ##############################################################################

# test_samples = []
# with gzip.open(sts_dataset_path, 'rt', encoding='utf8') as fIn:
#     reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
#     for row in reader:
#         if row['split'] == 'test':
#             score = float(row['score']) / 5.0 #Normalize score to range 0 ... 1
#             test_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=score))

# model = SentenceTransformer(model_save_path)
# test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, batch_size=train_batch_size, name='sts-test')
# test_evaluator(model, output_path=model_save_path)
