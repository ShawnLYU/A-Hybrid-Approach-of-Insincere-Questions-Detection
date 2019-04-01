from typing import *

from allennlp.data import Instance
from allennlp.data.fields import TextField, MetadataField, ArrayField
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter

from myutils import label_cols
from myutils import stats_path

from overrides import overrides

import numpy as np
import pandas as pd




@DatasetReader.register('quora_text_reader')
class QuoraDatasetReader(DatasetReader):
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy=False)
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.stats_train = pd.read_csv(stats_path+'/train_features.csv',index_col=0)
        self.stats_valid = pd.read_csv(stats_path+'/validation_features.csv',index_col=0)
        self.stats_test = pd.read_csv(stats_path+'/test_features.csv',index_col=0)
    @overrides
    def text_to_instance(self, tokens: List[Token],
                         row_id,
                         tag,
                         labels: np.ndarray=None) -> Instance:
        sentence_field = TextField(tokens, self.token_indexers)
        fields = {"tokens": sentence_field}
        if tag == 'train':
            stats = self.stats_train.loc[row_id].values
        elif tag == 'valid':
            stats = self.stats_valid.loc[row_id].values
        elif tag == 'test':
            stats = self.stats_test.loc[row_id].values

        fields["stats"] = ArrayField(array=stats)
        if labels is None:
            labels = np.array([0,0])
        label_field = ArrayField(array=labels)
        fields["label"] = label_field
        return Instance(fields)
    @overrides
    def _read(self, file_path: str) -> Iterator[Instance]:
        df = pd.read_csv(file_path,index_col=0)
        for i, row in df.iterrows():
            yield self.text_to_instance(
                [Token(x) for x in self.tokenizer(row["question_text"])], # Token is used to initialize a Token object
                row.name,
                row.tag,
                row[label_cols].values,
            )
    def tokenizer(self, x):
        return [w.text for w in SpacyWordSplitter(language='en', pos_tags=False).split_words(x)]



# from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
# from allennlp.data.token_indexers.elmo_indexer import ELMoCharacterMapper, ELMoTokenCharactersIndexer

# # data path
# training_data_path = '/Users/shawnlyu/Documents/projects/linguistics/CSC2511/Quora-Insincere-Questions-Classification/dev_data/filtered_train_data_all.csv'

# token_indexer = ELMoTokenCharactersIndexer()
# reader = QuoraDatasetReader(
#     token_indexers={"tokens": token_indexer}
# )

# train_ds = reader.read(training_data_path)

# vars(train_ds[0].fields["tokens"])





