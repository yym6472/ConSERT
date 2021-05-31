import torch
from torch import nn, Tensor
from typing import Union, Tuple, List, Iterable, Dict
from ..SentenceTransformer import SentenceTransformer
import logging


LARGE_NUM = 1e9

class MLP1(nn.Module):
    def __init__(self, hidden_dim=2048, norm=None, activation="relu"): # bottleneck structure
        super().__init__()
        ''' page 3 baseline setting
        Prediction MLP. The prediction MLP (h) has BN applied 
        to its hidden fc layers. Its output fc does not have BN
        (ablation in Sec. 4.4) or ReLU. This MLP has 2 layers. 
        The dimension of h’s input and output (z and p) is d = 2048, 
        and h’s hidden layer’s dimension is 512, making h a 
        bottleneck structure (ablation in supplement). 
        '''
        if activation == "relu":
            activation_layer = nn.ReLU()
        elif activation == "leakyrelu":
            activation_layer = nn.LeakyReLU()
        elif activation == "tanh":
            activation_layer = nn.Tanh()
        elif activation == "sigmoid":
            activation_layer = nn.Sigmoid()
        else:
            raise ValueError(f"Unknown activation function {hidden_activation}")
            
        if norm:
            if norm=='bn':
                norm_layer = nn.BatchNorm1d
            else: 
                norm_layer = nn.LayerNorm
                
            self.layer1 = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                norm_layer(hidden_dim),
                nn.ReLU(inplace=True)
            )
        else:
            self.layer1 = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True)
            )
                
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        """
        Adding BN to the output of the prediction MLP h does not work
        well (Table 3d). We find that this is not about collapsing. 
        The training is unstable and the loss oscillates.
        """

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x 
    
class SimCLRLoss(nn.Module):
    """
    This loss was used in our SBERT publication (https://arxiv.org/abs/1908.10084) to train the SentenceTransformer
    model on NLI data. It adds a softmax classifier on top of the output of two transformer networks.

    :param model: SentenceTransformer model
    :param sentence_embedding_dimension: Dimension of your sentence embeddings
    :param num_labels: Number of different labels
    :param concatenation_sent_rep: Concatenate vectors u,v for the softmax classifier?
    :param concatenation_sent_difference: Add abs(u-v) for the softmax classifier?
    :param concatenation_sent_multiplication: Add u*v for the softmax classifier?

    Example::

        from sentence_transformers import SentenceTransformer, SentencesDataset, losses
        from sentence_transformers.readers import InputExample

        model = SentenceTransformer('distilbert-base-nli-mean-tokens')
        train_examples = [InputExample(InputExample(texts=['First pair, sent A', 'First pair, sent B'], label=0),
            InputExample(texts=['Second Pair, sent A', 'Second Pair, sent B'], label=3)]
        train_dataset = SentencesDataset(train_examples, model)
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
        train_loss = losses.SoftmaxLoss(model=model, sentence_embedding_dimension=model.get_sentence_embedding_dimension(), num_labels=train_num_labels)
    """
    def __init__(self,
                 model: SentenceTransformer,
                 sentence_embedding_dimension: int,
                 num_labels: int,
                 concatenation_sent_rep: bool = True,
                 concatenation_sent_difference: bool = True,
                 concatenation_sent_multiplication: bool = False,
                 concatenation_sent_max_square: bool = False,           # 拼接两个句子表示的max-square（如寐建议的一个trick）
                 data_augmentation_strategy: str = "normal",               # 数据增强策略，可选项：不进行增强“none”、对抗“adv”、mean和max pooling对比“meanmax”、TODO
                 projection_norm_type: str = "ln",
                 do_hidden_normalization: bool = True,                  # 进行对比损失之前，是否对句子表示做正则化
                 temperature: float = 1.0,                              # 对比损失中的温度系数，仅对于交叉熵损失有效
                 mapping_to_small_space: int = None,                    # 是否将句子表示映射到一个较小的向量空间进行对比损失（类似SimCLR），及其映射的最终维度
                 add_contrastive_predictor: bool = True,                 # 是否在对比学习中，将句子表示非线性映射到同等维度（类似SimSiam），以及将其添加到哪一端（normal or adv）
                 projection_hidden_dim: int = None,                     # 定义MLP的中间维度大小，对于上面两个选项（mapping & predictor）均有用
                 projection_use_batch_norm: bool = None,                # 定义是否在MLP的中间层添加BatchNorm，对于上面两个选项（mapping & predictor）均有用
                ):
        super(SimCLRLoss, self).__init__()
        self.model = model
        self.num_labels = num_labels
        self.concatenation_sent_rep = concatenation_sent_rep
        self.concatenation_sent_difference = concatenation_sent_difference
        self.concatenation_sent_multiplication = concatenation_sent_multiplication
        self.concatenation_sent_max_square = concatenation_sent_max_square
        
        self.data_augmentation_strategy = data_augmentation_strategy
        self.do_hidden_normalization = do_hidden_normalization
        self.temperature = temperature
        self.add_contrastive_predictor = add_contrastive_predictor
        if add_contrastive_predictor:
            self.predictor = MLP1(hidden_dim=sentence_embedding_dimension, norm=projection_norm_type)

        num_vectors_concatenated = 0
        if concatenation_sent_rep:
            num_vectors_concatenated += 2
        if concatenation_sent_difference:
            num_vectors_concatenated += 1
        if concatenation_sent_multiplication:
            num_vectors_concatenated += 1
        if concatenation_sent_max_square:
            num_vectors_concatenated += 1
            
        logging.info("Softmax loss: #Vectors concatenated: {}".format(num_vectors_concatenated))
        self.classifier = nn.Linear(num_vectors_concatenated * sentence_embedding_dimension, num_labels)
 
    def _reps_to_output(self, rep_a: torch.Tensor, rep_b: torch.Tensor):
        vectors_concat = []
        if self.concatenation_sent_rep:
            vectors_concat.append(rep_a)
            vectors_concat.append(rep_b)

        if self.concatenation_sent_difference:
            vectors_concat.append(torch.abs(rep_a - rep_b))

        if self.concatenation_sent_multiplication:
            vectors_concat.append(rep_a * rep_b)
        
        if self.concatenation_sent_max_square:
            vectors_concat.append(torch.max(rep_a, rep_b).pow(2))

        features = torch.cat(vectors_concat, 1)

        output = self.classifier(features)
        return output
    
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
        if self.data_augmentation_strategy == "meanmax":  # 使用mean-max pooling的对比
            rep_dicts = [self.model(sentence_feature) for sentence_feature in sentence_features]
            reps_mean = [rep_dict['pad_mean_tokens'] for rep_dict in rep_dicts]
            reps_max = [rep_dict['pad_max_tokens'] for rep_dict in rep_dicts]
            rep_a_view1, rep_a_view2 = reps_mean[0], reps_max[0]

        elif self.data_augmentation_strategy == "normal":  # 最原始的版本，只需获取rep_a和rep_b即可  # TODO: 在这里添加更多的数据增强策略
            reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
            rep_a = reps[0]
            rep_a_view1, rep_a_view2 = rep_a, rep_a
        else:
            raise ValueError("Invalid data augmentation strategy")
        
        # add predictor
        if self.add_contrastive_predictor:
            rep_a_view1 = self.predictor(rep_a_view1)
            rep_a_view2 = self.predictor(rep_a_view2)
        
        final_loss = 0
        
        contrastive_loss = self._contrastive_loss_forward(rep_a_view1, rep_a_view2, hidden_norm=self.do_hidden_normalization, temperature=self.temperature)
        self.model.tensorboard_writer.add_scalar(f"train_contrastive_loss", contrastive_loss.item(), global_step=self.model.global_step)
        final_loss += contrastive_loss
        self.model.tensorboard_writer.add_scalar(f"train_contrastive_loss_total", contrastive_loss.item(), global_step=self.model.global_step)
        
        return final_loss