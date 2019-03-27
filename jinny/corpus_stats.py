import pandas as pd
import numpy as np
import spacy
import string
from scipy import stats
from matplotlib import pyplot as plt
from progress.bar import Bar

# filepath = '../data/mytrain.csv'
filepath = '../data/filtered_train_data_all.csv'
# filepath = 'toy_set.csv' # a small set of 2000 questions for testing
df_data = pd.read_csv(filepath)

nlp = spacy.load('en_core_web_sm')

PUNCT_DICT = {'all_punctuation': string.punctuation, 'commas': ',', \
'periods': '.', 'quotation_marks': '\'\"', 'question_marks': '?', \
'exclamation_marks': '!', 'other_punctuations': [s for s in string.punctuation if s not in ',.\'\"?!']}

POS_LIST = ['ADJ', 'ADV', 'INTJ', 'NOUN', 'PROPN', 'VERB', 'ADP', 'AUX', \
'CCONJ', 'DET', 'NUM', 'PART', 'PRON', 'SCONJ', 'SYM', 'X']
# Reference: https://universaldependencies.org/u/pos/

ENT_LIST = ['PERSON', 'NORP', 'FAC', 'ORG', 'GPE', 'LOC', 'PRODUCT', 'EVENT', \
'WORK_OF_ART', 'LAW', 'LANGUAGE', 'DATE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY',
'ORDINAL', 'CARDINAL']
#reference: https://spacy.io/api/annotation#section-named-entities

def build_count_dict(sentence):
	""" Return count dictionaries for sentence mapping from labels to the count of words
	that satisfies a corresponding condition: 
	1 - char in char_type (punctuation)
	2 - taken.pos_ == pos (part of speech)
	3 - ent.label_ == ent (named entites)
	"""
	punc = {key: 0 for key in PUNCT_DICT.keys()}
	pos = {pos: 0 for pos in POS_LIST}
	ent = {ent: 0 for ent in ENT_LIST}

	doc = nlp(sentence)
	ents = doc.ents

	for word in sentence:
		for key, value in PUNCT_DICT.items():
			if word in value:
				punc[key] += 1

	for token in doc:
		if token.pos_ in POS_LIST:
			pos[token.pos_] += 1

	for e in ents:
		if e.label_ in ENT_LIST:
			ent[e.label_] += 1

	return punc, pos, ent


def data_collection(dataframe):
	""" Return statistical data of sentences with label, which is 0 for negative
	and 1 for positive. 
	"""
	sentences = dataframe['question_text'].values
	# punctuations
	punc = dict((key, []) for key in PUNCT_DICT.keys())
	# punc_count = dict((key, 0) for key in PUNCT_DICT.keys())
	pos = dict((pos, []) for pos in POS_LIST)
	# pos_count = dict((pos, 0) for pos in POS_LIST)
	ent = dict((ent, []) for ent in ENT_LIST)

	data_container = [punc, pos, ent]

	bar = Bar("Collecting data over sentences", max=len(sentences))
	for s in sentences:
		# punctuations
		punc_dict, pos_dict, ent_dict = build_count_dict(s)

		data = [punc_dict, pos_dict, ent_dict]

		for i in range(len(data)):
			for key, value in data[i].items():
				data_container[i][key].append(value)

		bar.next()
	bar.finish()

	# def sum_values(d):
	# 	# a helper function for summing values for each key in d
	# 	return {key: sum(d[key]) for key in d.keys()}

	# punc_count = sum_values(punc)
	# punc_count.pop('all_punctuation') # Remove all_punctuations
	# pos_count = sum_values(pos)

	for container in data_container:
		for key, value in container.items():
			dataframe[key] = pd.Series(value, index=dataframe.index)

def ks_test(set1, set2, theme):
	""" Conduct KS test to compare set1 and set2. Print the results and return
	True iff set1 and set2 are significantly different at 0.001 level.  

	Theme is a text label for the comparison. 
	"""

	ks_test_score, ks_p_value = stats.ks_2samp(set1, set2)

	print("===== KS test for {} =====".format(theme))

	print("KS statistic: {}\np-value: {}".format(ks_test_score, ks_p_value))

	# Since it is a two-tailed test, the difference is considered significant
	# when p value is smaller thatn 0.01
	if ks_p_value < 0.01:
		print("The two distributions are significantly different. ")
		return True

	return False

def main():
	# Getting raw data from data_collection function
	data_collection(df_data)
	df_positive, df_negative = df_data[df_data['target']==1], df_data[df_data['target'] == 0]

	# Containers for punctuation marks/PoS/entities counts
	features = {'punctuation':[PUNCT_DICT.keys(), []], \
	'pos_tag':[POS_LIST, []], 'ent':[ENT_LIST, []]}

	for key, value in features.items():
		for label in value[0]:
			if ks_test(df_positive[label].values, df_negative[label].values, label):
				value[1].append(label)
		df = df_data[value[1]]
		df['target'] = df_data['target']
		filename = '{}.csv'.format(key)
		df.to_csv(filename, index=0)

	for key, value in features.items():
		print('{} test results: {}'.format(key, value[1]))

	# Begin analyzing punctuation 
	# 1. Distributions of punctuation in positive and negative samples
	# all_punc_norm_df = pd.DataFrame(data={
	# 	'positive': pd.Series(pos_punc['all_punctuation']).value_counts(normalize=True), 
	# 	'negative': pd.Series(neg_punc['all_punctuation']).value_counts(normalize=True)
	# 	})

	# all_punc_norm_df.plot(kind='bar', title="Punctuation Density Distributions")

	# 2. Distribution of different types of punctuation

	# TODO: Pie plots are not considered as a good data visualization method. 
	# This is a convenient compromised choice. 
	# If necessary, consider following alternatives: 
	# http://www.storytellingwithdata.com/blog/2014/06/alternatives-to-pies
	# plt.figure('puncpiecharts')
	# plt.subplot(121, title='Positive Samples')
	# labels = list(pos_punc_count.keys())
	# counts = list(pos_punc_count.values())
	# plt.pie(counts, labels=labels, autopct='%1.1f%%')

	# plt.subplot(122, title='Negative Samples')
	# labels = list(neg_punc_count.keys())
	# counts = list(neg_punc_count.values())
	# plt.pie(counts, labels=labels, autopct='%1.1f%%')

if __name__ == '__main__':
	main()
	# plt.show()