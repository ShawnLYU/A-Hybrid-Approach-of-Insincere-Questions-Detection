from typing import *
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
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder

from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.nn import util as nn_util
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.data.iterators import BucketIterator
from allennlp.training.trainer import Trainer
from allennlp.predictors import SentenceTaggerPredictor

from allennlp.data.iterators import DataIterator
from tqdm import tqdm
from scipy.special import expit # the sigmoid function
from allennlp.models import Model

import numpy as np

def tonp(tsr): return tsr.detach().cpu().numpy()
 
from allennlp.models.archival import Archive, load_archive
class Predictor:
    def __init__(self, archive: Archive, iterator: DataIterator,
                 cuda_device: int) -> None:
        self.model = archive.model
        self.iterator = iterator
        self.cuda_device = cuda_device
    def _extract_data(self, batch) -> np.ndarray:
        out_dict = self.model(**batch)
        return expit(tonp(out_dict["class_logits"]))
    def predict(self, ds: Iterable[Instance]) -> np.ndarray:
        pred_generator = self.iterator(ds, num_epochs=1, shuffle=False)
        self.model.eval()
        pred_generator_tqdm = tqdm(pred_generator,
                                   total=self.iterator.get_num_batches(ds))
        preds = []
        with torch.no_grad():
            for batch in pred_generator_tqdm:
                # batch = nn_util.move_to_device(batch, self.cuda_device)
                preds.append(self._extract_data(batch))
        return np.concatenate(preds, axis=0)



from packages import model
from packages import dataset_reader
from allennlp.models.archival import load_archive
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.dataset_readers import DatasetReader

from allennlp.data.token_indexers.elmo_indexer import ELMoTokenCharactersIndexer


USE_GPU = torch.cuda.is_available()

from allennlp.data.iterators import BasicIterator
# iterate over the dataset without changing its order
seq_iterator = BasicIterator(batch_size=64)
seq_iterator.index_with(Vocabulary())


Archive_ds = DatasetReader.by_name('quora_text_reader')
reader = Archive_ds(token_indexers={"elmo": ELMoTokenCharactersIndexer()})
training_data_path = '/u/shawnlyu/projects/linguistics/workdir/dev_data/filtered_train_data_all.csv'
train_ds = reader.read(training_data_path)


predictor = Predictor(archive, seq_iterator, cuda_device=0 if USE_GPU else -1)
train_preds = predictor.predict(train_ds) 

np.argmax(train_preds,axis=1)

# tokens = batch["tokens"]
# labels = batch
# mask = get_text_field_mask(tokens)

# archive = load_archive('res/model.tar.gz')
# model = archive.model


# config = archive.config.duplicate()
# dataset_reader_params = config["dataset_reader"]
# dataset_reader = DatasetReader.from_params(dataset_reader_params)



# embeddings = model.word_embeddings(tokens)


# state = model.encoder(embeddings, mask)
# class_logits = model.hidden(state)
# class_logits



# test_preds = predictor.predict(test_ds) 

