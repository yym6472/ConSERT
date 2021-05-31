import torch
from torch import nn, Tensor
from typing import Union, Tuple, List, Iterable, Dict
from sentence_transformers import SentenceTransformer

LARGE_NUM = 1e9

class ShuffleCLLoss(nn.Module):
    def __init__(self,
                 model: SentenceTransformer,
                 sentence_embedding_dimension: int,
                 contrastive_loss_rate: float = 1.0,                    # 对比损失的系数
                 temperature: float = 1.0):                             # 对比损失中的温度系数，仅对于交叉熵损失有效
        super(ShuffleCLLoss, self).__init__()
        self.model = model
        
        self.contrastive_loss_rate = contrastive_loss_rate
        self.temperature = temperature
    
    def _contrastive_loss_forward(self,
                                  hidden1: torch.Tensor,
                                  hidden2: torch.Tensor,
                                  hidden_norm: bool = True,
                                  temperature: float = 1.0):
        """
        hidden1/hidden2: (bsz, dim)
        """
        batch_size, hidden_dim = hidden1.shape
        
        if hidden_norm:
            hidden1 = torch.nn.functional.normalize(hidden1, p=2, dim=-1)
            hidden2 = torch.nn.functional.normalize(hidden2, p=2, dim=-1)

        hidden1_large = hidden1
        hidden2_large = hidden2
        labels = torch.arange(0, batch_size).to(device=hidden1.device)
        masks = torch.nn.functional.one_hot(torch.arange(0, batch_size), num_classes=batch_size).to(device=hidden1.device, dtype=torch.float)

        logits_aa = torch.matmul(hidden1, hidden1_large.transpose(0, 1)) / temperature  # shape (bsz, bsz)
        logits_aa = logits_aa - masks * LARGE_NUM
        logits_bb = torch.matmul(hidden2, hidden2_large.transpose(0, 1)) / temperature  # shape (bsz, bsz)
        logits_bb = logits_bb - masks * LARGE_NUM
        logits_ab = torch.matmul(hidden1, hidden2_large.transpose(0, 1)) / temperature  # shape (bsz, bsz)
        logits_ba = torch.matmul(hidden2, hidden1_large.transpose(0, 1)) / temperature  # shape (bsz, bsz)

        loss_a = torch.nn.functional.cross_entropy(torch.cat([logits_ab, logits_aa], dim=1), labels)
        loss_b = torch.nn.functional.cross_entropy(torch.cat([logits_ba, logits_bb], dim=1), labels)
        loss = loss_a + loss_b
        return loss

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        labels = None  # 保证不使用labels
        
        if not self.training:  # 验证阶段或预测阶段
            reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
            return reps, None
        else:  # 训练阶段
            # 生成增强版本
            sentence_feature = sentence_features[0]
            ori_feature_keys = set(sentence_feature.keys())  # record the keys since the features will be updated
            rep = self.model(sentence_feature)['sentence_embedding']
            sentence_feature = {k: v for k, v in sentence_feature.items() if k in ori_feature_keys}
            self.model[0].auto_model.set_flag("data_aug_shuffle", True)
            rep_shuffle = self.model(sentence_feature)['sentence_embedding']

            contrastive_loss = self._contrastive_loss_forward(rep, rep_shuffle, hidden_norm=True, temperature=self.temperature)
            self.model.tensorboard_writer.add_scalar(f"train_contrastive_loss", contrastive_loss.item(), global_step=self.model.global_step)
            final_loss = self.contrastive_loss_rate * contrastive_loss
            self.model.tensorboard_writer.add_scalar(f"final_loss", final_loss.item(), global_step=self.model.global_step)
            return final_loss
        

from training_nli import set_seed
from torch.utils.data import DataLoader
import math
from sentence_transformers import models, losses
from sentence_transformers import SentencesDataset, SentenceTransformer, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
import os
import gzip
import csv
import shutil

from tensorboardX import SummaryWriter
from eval import eval_nli_unsup
from correlation_visualization import corr_visualization

set_seed(1, for_multi_gpu=False)

nli_dataset_path = 'datasets/AllNLI_texts.txt'
sts_dataset_path = 'datasets/stsbenchmark.tsv.gz'

train_batch_size = 32
num_epochs = 1
learning_rate = 5e-7
evaluation_steps = 200
cl_rate = 0.15
cl_temperature = 0.1
attention_probs_dropout_prob = 0.0
hidden_dropout_prob = 0.0
model_save_path = "./output/reproduce-shuffle-bsz32-nodropout"
if os.path.exists(model_save_path):
    shutil.rmtree(model_save_path)
os.mkdir(model_save_path)

# build model
word_embedding_model = models.Transformer("./bert-base-uncased", attention_probs_dropout_prob=attention_probs_dropout_prob, hidden_dropout_prob=hidden_dropout_prob)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=True,
                               pooling_mode_cls_token=False,
                               pooling_mode_max_tokens=False)
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
model.tensorboard_writer = SummaryWriter(os.path.join(model_save_path, "logs"))

with open(nli_dataset_path, 'r', encoding='utf8') as f:
    texts = [line.strip() for line in f if line.strip()]
train_samples = [InputExample(texts=[text]) for text in texts]
train_dataset = SentencesDataset(train_samples, model=model)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)

train_loss = ShuffleCLLoss(model=model, sentence_embedding_dimension=model.get_sentence_embedding_dimension(), contrastive_loss_rate=cl_rate, temperature=cl_temperature)

dev_samples = []
with gzip.open(sts_dataset_path, 'rt', encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        if row['split'] == 'dev':
            score = float(row['score']) / 5.0 #Normalize score to range 0 ... 1
            dev_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=score))
dev_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, batch_size=train_batch_size, name='sts-dev')

warmup_steps = math.ceil(len(train_dataset) * num_epochs / train_batch_size * 0.1) #10% of train data for warm-up
print("Warmup-steps: {}".format(warmup_steps))

# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=dev_evaluator,
          epochs=num_epochs,
          optimizer_params={'lr': learning_rate, 'eps': 1e-6, 'correct_bias': False},
          evaluation_steps=evaluation_steps,
          warmup_steps=warmup_steps,
          output_path=model_save_path,
          use_apex_amp=True,
          apex_amp_opt_level="O1")

# Test on STS Benchmark
test_samples = []
with gzip.open(sts_dataset_path, 'rt', encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        if row['split'] == 'test':
            score = float(row['score']) / 5.0 #Normalize score to range 0 ... 1
            test_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=score))
model = SentenceTransformer(model_save_path)
test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, batch_size=train_batch_size, name='sts-test')
test_evaluator(model, output_path=model_save_path)

# Test on unsupervised dataset (mainly STS related dataset)
eval_nli_unsup(model_save_path, main_similarity=None)
eval_nli_unsup(model_save_path, main_similarity=None, last2avg=True)
corr_visualization(model_save_path)