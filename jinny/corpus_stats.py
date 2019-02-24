import pandas as pdimport numpy as npimport spacyimport stringfrom scipy import statsfrom matplotlib import pyplot as pltfrom progress.bar import Bar# filepath = '../data/mytrain.csv'filepath = 'toy_set.csv' # a small set of 2000 questions for testingdf_data = pd.read_csv(filepath)nlp = spacy.load('en_core_web_sm')# Spliting data into two sets based on target valuedf_positive, df_negative = df_data[df_data['target']==1], df_data[df_data['target'] == 0]# Getting sentences in each datasetsen_pos, sen_neg = df_positive['question_text'].values, df_negative['question_text'].values# helper function to count the number of certain types of characters # (e.g. punctuation)count_char = lambda sentence, char_type: sum([1 for char in sentence if char in char_type])PUNCT_DICT = {'all_punctuation': string.punctuation, 'commas': ',', \'periods': '.', 'quotation_marks': '\'\"', 'question_marks': '?', \'exclamation_marks': '!', 'other_punctuations': [s for s in string.punctuation if s not in ',.\'\"?!']}# Return a dictionary of the number of each punctuation mark a sentence.build_punc_dict = lambda sentence: {key: count_char(sentence, value) for key, value in PUNCT_DICT.items()}# Helper function: count the number of a certain type of PoS in a sentencecount_pos = lambda sentence, pos: sum([1 for token in nlp(sentence) if token.pos_ == pos])# Following is the list of all universal POS tags except 'PUNCT'POS_LIST = ['ADJ', 'ADV', 'INTJ', 'NOUN', 'PROPN', 'VERB', 'ADP', 'AUX', \'CCONJ', 'DET', 'NUM', 'PART', 'PRON', 'SCONJ', 'SYM', 'X']# Reference: https://universaldependencies.org/u/pos/# Return a dictionary of the number of each part of speech for a sentence.build_pos_dict = lambda sentence: {pos: count_pos(sentence, pos) for pos in POS_LIST}def data_collection(sentences):	""" Return statistical data of sentences with label, which is 0 for negative	and 1 for positive. 	"""	# punctuations	punc = dict((key, []) for key in PUNCT_DICT.keys())	punc_count = dict((key, 0) for key in PUNCT_DICT.keys())	pos = dict((pos, []) for pos in POS_LIST)	pos_count = dict((pos, 0) for pos in POS_LIST)	bar = Bar("Collecting data over sentences", max=len(sentences))	for s in sentences:		# punctuations		punc_dict = build_punc_dict(s)		pos_dict = build_pos_dict(s)		for key, value in punc_dict.items():			punc[key].append(value)		for key, value in pos_dict.items():			pos[key].append(value)		bar.next()	bar.finish()	def sum_values(d):		# a helper function for summing values for each key in d		return {key: sum(d[key]) for key in d.keys()}	punc_count = sum_values(punc)	punc_count.pop('all_punctuation') # Remove all_punctuations	pos_count = sum_values(pos)	return punc, punc_count, pos, pos_countdef ks_test(set1, set2, theme):	""" Conduct KS test to compare set1 and set2. Print the results and return	True iff set1 and set2 are significantly different at 0.001 level.  	Theme is a text label for the comparison. 	"""	ks_test_score, ks_p_value = stats.ks_2samp(set1, set2)	print("===== KS test for {} =====".format(theme))	print("KS statistic: {}\np-value: {}".format(ks_test_score, ks_p_value))	# Since it is a two-tailed test, the difference is considered significant	# when p value is smaller thatn 0.01	if ks_p_value < 0.01:		print("The two distributions are significantly different. ")		return True	return Falsedef main():	# Getting raw data from data_collection function	pos_punc, pos_punc_count, pos_pos, pos_pos_count = data_collection(sen_pos)	neg_punc, neg_punc_count, neg_pos, pos_pos_count = data_collection(sen_neg)	# Containers for punctuation marks/PoS of which distributions are 	# significantly different between positive and negative samples	important_punc = []	important_pos = []	# Begin analyzing punctuation 	# 1. Distributions of punctuation in positive and negative samples	all_punc_norm_df = pd.DataFrame(data={		'positive': pd.Series(pos_punc['all_punctuation']).value_counts(normalize=True), 		'negative': pd.Series(neg_punc['all_punctuation']).value_counts(normalize=True)		})	all_punc_norm_df.plot(kind='bar', title="Punctuation Density Distributions")	for key in pos_punc:		if ks_test(pd.Series(pos_punc[key]).values, 			pd.Series(neg_punc[key]).values, key):			important_punc.append(key)	# 2. Distribution of different types of punctuation	# TODO: Pie plots are not considered as a good data visualization method. 	# This is a convenient compromised choice. 	# If necessary, consider following alternatives: 	# http://www.storytellingwithdata.com/blog/2014/06/alternatives-to-pies	# plt.figure('puncpiecharts')	# plt.subplot(121, title='Positive Samples')	# labels = list(pos_punc_count.keys())	# counts = list(pos_punc_count.values())	# plt.pie(counts, labels=labels, autopct='%1.1f%%')	# plt.subplot(122, title='Negative Samples')	# labels = list(neg_punc_count.keys())	# counts = list(neg_punc_count.values())	# plt.pie(counts, labels=labels, autopct='%1.1f%%')		# Begin Analyzing PoS	for key in pos_pos:		if ks_test(pd.Series(pos_pos[key]).values, pd.Series(neg_pos[key]).values, key):			important_pos.append(key)	print("Punctuation test results: {}".format(' '.join(important_punc)))	print("PoS test results: {}".format(' '.join(important_pos)))if __name__ == '__main__':	main()	# plt.show()