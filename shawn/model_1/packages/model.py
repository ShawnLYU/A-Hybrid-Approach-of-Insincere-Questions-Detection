from typing import Iterator, List, Dict
import torch
import torch.optim as optim
import numpy as np
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

torch.manual_seed(1)



@Model.register('quora_insincere_classification')
class BaselineModel(Model):
    def __init__(self,
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2VecEncoder,
                 vocab: Vocabulary) -> None:
        super().__init__(vocab)
        self.word_embeddings = text_field_embedder
        self.encoder = encoder
        self.hidden = torch.nn.Linear(self.encoder.get_output_dim(), len(label_cols))
        # self.output = torch.nn.Sigmoid()
        # This loss combines a `Sigmoid` layer and the `BCELoss` in one single class
        # self.accuracy = torch.nn.BCEWithLogitsLoss()
        self.loss = torch.nn.BCEWithLogitsLoss()
    def forward(self,
                tokens: Dict[str, torch.Tensor],
                label: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        mask = get_text_field_mask(tokens)
        embeddings = self.word_embeddings(tokens)
        encoder_out = self.encoder(embeddings, mask)
        cls_logits = self.hidden(encoder_out)
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












