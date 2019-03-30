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

from sklearn.metrics import precision_recall_curve


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
        labels = []
        with torch.no_grad():
            for batch in pred_generator_tqdm:
                labels.append(batch['label'])
                preds.append(self._extract_data(batch))
        # for e in preds[0]:
        #     print("aaaaaaaaaaaaaa",len(e))    
        return np.concatenate(labels, axis=0),np.concatenate(preds, axis=0)



import argparse
parser = argparse.ArgumentParser(description='Examples: python predicator path/to/file 1.')
parser.add_argument("path")
parser.add_argument("model_path")
parser.add_argument("label")
args = parser.parse_args()
predict_label = ''
if args.label == 1:
    predict_label = 'train'
elif args.label == 2:
    predict_label = 'validation'
elif args.label == 3:
    predict_label = 'test'







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
data_path = args.path
train_ds = reader.read(data_path)

archive = load_archive(args.model_path+'/model.tar.gz')
model = archive.model

predictor = Predictor(archive, seq_iterator, cuda_device=0 if USE_GPU else -1)
labels, train_preds = predictor.predict(train_ds) 

# predictions = np.argmax(train_preds,axis=1)
# targets = np.argmax(labels,axis=1)
predictions = np.amax(train_preds,axis=1)
targets = np.amax(labels,axis=1)



from sklearn.metrics import precision_recall_curve
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from sklearn.utils.fixes import signature

precision, recall, thresholds = precision_recall_curve(y_true=targets,probas_pred=predictions)
np.savetxt('predictions.csv',np.array(predictions))
np.savetxt('targets.csv',np.array(targets))
np.savetxt('precision.csv',np.array(precision))
np.savetxt('recall.csv',np.array(recall))
np.savetxt('thresholds.csv',np.array(thresholds))

# In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
step_kwargs = ({'step': 'post'}
               if 'step' in signature(plt.fill_between).parameters
               else {})
plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
          average_precision))
plt.savefig('p_r_'+predict_label+'.png')
















