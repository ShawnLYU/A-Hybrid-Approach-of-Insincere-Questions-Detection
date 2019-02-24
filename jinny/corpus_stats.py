import pandas as pd
import numpy as np
import spacy
import string
from scipy import stats
from matplotlib import pyplot as plt
from progress.bar import Bar

# filepath = '../data/mytrain.csv'
filepath = 'toy_set.csv' # a small set of 2000 questions for testing
df_data = pd.read_csv(filepath)

nlp = spacy.load('en_core_web_sm')

# helper function to count the number of certain types of characters 
# (e.g. punctuation)
count_char = lambda sentence, char_type: sum([1 for char in sentence if char in char_type])

PUNCT_DICT = {'all_punctuation': string.punctuation, 'commas': ',', \
'periods': '.', 'quotation_marks': '\'\"', 'question_marks': '?', \
'exclamation_marks': '!', 'other_punctuations': [s for s in string.punctuation if s not in ',.\'\"?!']}

# Return a dictionary of the number of each punctuation mark a sentence.
build_punc_dict = lambda sentence: {key: count_char(sentence, value) for key, value in PUNCT_DICT.items()}

# Helper function: count the number of a certain type of PoS in a sentence
count_pos = lambda sentence, pos: sum([1 for token in nlp(sentence) if token.pos_ == pos])

# Following is the list of all universal POS tags except 'PUNCT'
POS_LIST = ['ADJ', 'ADV', 'INTJ', 'NOUN', 'PROPN', 'VERB', 'ADP', 'AUX', \
'CCONJ', 'DET', 'NUM', 'PART', 'PRON', 'SCONJ', 'SYM', 'X']
# Reference: https://universaldependencies.org/u/pos/

# Return a dictionary of the number of each part of speech for a sentence.
build_pos_dict = lambda sentence: {pos: count_pos(sentence, pos) for pos in POS_LIST}

# Similar to POS, the following code is for named entities
count_ent = lambda sentence, label: sum([1 for ent in nlp(sentence).ents if ent.label_ == label])

ENT_LIST = ['PERSON', 'NORP', 'FAC', 'ORG', 'GPE', 'LOC', 'PRODUCT', 'EVENT', \
'WORK_OF_ART', 'LAW', 'LANGUAGE', 'DATE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY',
'ORDINAL', 'CARDINAL']
#reference: https://spacy.io/api/annotation#section-named-entities

build_ent_dict = lambda sentence: {ent: count_ent(sentence, ent) for ent in ENT_LIST}

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
		punc_dict = build_punc_dict(s)
		pos_dict = build_pos_dict(s)
		ent_dict = build_ent_dict(s)

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

	# Containers for punctuation marks/PoS/entities of which distributions are 
	# significantly different between positive and negative samples
	features = {'punctuation':[PUNCT_DICT.keys(), []], \
	'pos_tag':[POS_LIST, []], 'ent':[ENT_LIST, []]}

	for key, value in features.items():
		for label in value[0]:
			if ks_test(df_positive[label].values, df_negative[label].values, label):
				value[1].append(label)
		df = df_data[value[1]]
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