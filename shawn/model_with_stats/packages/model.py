# allennlp train config.json -s res --include-package packages --force
from typing import Iterator, List, Dict
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from allennlp.data import Instance
from allennlp.data.fields import TextField, SequenceLabelField
from allennlp.data.dataset_readers import DatasetReader
from allennlp.common.file_utils import cached_path
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
# from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder, PytorchSeq2SeqWrapper
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder

from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.data.iterators import BucketIterator
from allennlp.training.trainer import Trainer
from allennlp.predictors import SentenceTaggerPredictor

from myutils import label_cols

from myutils import stats_path
from myutils import device
from myutils import stats_dim

torch.manual_seed(1)



@Model.register('quora_insincere_classification')
class BaselineModel(Model):
    def __init__(self,
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2VecEncoder,
                 encoder_cnn: Seq2VecEncoder,
                 vocab: Vocabulary) -> None:
        super().__init__(vocab)
        self.word_embeddings = text_field_embedder
        self.encoder = encoder
        # self.encoder_cnn = encoder_cnn
        
        self.encoder_cnn = torch.nn.Conv1d(in_channels=1,out_channels=64,kernel_size=2)
        self.max_pooling = torch.nn.MaxPool1d(kernel_size=127, stride=1, padding=0)
        self.hidden = torch.nn.Linear(64+stats_dim, len(label_cols))


        self.stats_train = pd.read_csv(stats_path+'/train_features.csv',index_col=0)
        self.stats_valid = pd.read_csv(stats_path+'/validation_features.csv',index_col=0)
        self.stats_test = pd.read_csv(stats_path+'/test_features.csv',index_col=0)

        # self.output = torch.nn.Sigmoid()
        # This loss combines a `Sigmoid` layer and the `BCELoss` in one single class
        # self.accuracy = torch.nn.BCEWithLogitsLoss()
        self.loss = torch.nn.BCEWithLogitsLoss()
    def forward(self,
                tokens: Dict[str, torch.Tensor],
                token_id,
                tag,
                label: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        # embeddings
        # print('token_id',token_id)
        mask = get_text_field_mask(tokens)
        embeddings = self.word_embeddings(tokens)
        N = embeddings.shape[0]
        # print('embeddings',embeddings.shape)
        # bi-LSTM
        encoder_after_lstm = self.encoder(embeddings, mask)
        # print('encoder_after_lstm',encoder_after_lstm.shape)
        # CNN
        encoder_after_cnn = self.encoder_cnn(encoder_after_lstm.view(N,1,128))
        # print('encoder_after_cnn',encoder_after_cnn.shape)
        encoder_after_pooling = self.max_pooling(encoder_after_cnn)
        # print('encoder_after_pooling',encoder_after_pooling.shape)
        
        encoder_after_pooling = torch.squeeze(encoder_after_pooling,2)
        # print('reshape',encoder_after_pooling.shape)
            
        # concatenate
        stats_tensor = torch.FloatTensor().to(device)
        if tag[0] == 'train':
            stats_tensor = torch.FloatTensor(self.stats_train.loc[token_id].values).to(device)
        elif tag[0] == 'validation':
            stats_tensor = torch.FloatTensor(self.stats_valid.loc[token_id].values).to(device)
        elif tag[0] == 'test':
            stats_tensor = torch.FloatTensor(self.stats_test.loc[token_id].values).to(device)
        dense = torch.cat((encoder_after_pooling,stats_tensor),dim=1) # concatenate horizontally
        # print('dense',dense.shape)


        # DNN

        cls_logits = self.hidden(dense)
        # print('cls_logits',cls_logits.shape)
        # print(cls_logits)
        # res = self.output(cls_logits)
        # output = {"res": cls_logits, "prediction": np.argmax(cls_logits,axis=0)}
        output = {"class_logits": cls_logits}
        if label is not None:
            # self.accuracy(tag_logits, label, mask)
            output["loss"] = self.loss(cls_logits, label)
        return output
    # def get_metrics(self, reset: bool = False) -> Dict[str, float]:
    #     return {"accuracy": self.accuracy.get_metric(reset)}












