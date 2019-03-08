import pandas as pd
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from progress.bar import Bar

# filepath = '../data/filtered_train_data_all.csv'
filepath = 'toy_set.csv'

df_data = pd.read_csv(filepath, index_col=0)

stopwords = set(stopwords.words('english'))
punctuation = string.punctuation

def sentence_sentiment(s):
	testimonial = TextBlob(s)
	# The polarity score is a float within the range [-1.0, 1.0].
	# The subjectivity is a float within the range [0.0, 1.0] where 0.0 is very objective and 1.0 is very subjective.
	return testimonial.sentiment.polarity, testimonial.sentiment.subjectivity

def stop_word_removal(s):
	tokens = word_tokenize(s)
	filtered = [w for w in tokens if w not in stopwords and w not in punctuation]
	return ' '.join(filtered)

def sentence_processing(df):
	""" This function performs two tasks: 
	- Sentiment extraction and write the results into a csv file
	- Write filtered words into two text files, one for labelled data, one for unlabelled data
	"""
	pos_df, neg_df = df[df['target']==1], df[df['target']==0]
	sentences, targets = df['question_text'].values, df['target'].values
	sentiment_dict = {'sentiment':[], 'polarity':[]}
	for i in len(sentences):
		sentence = sentences[i]
		target = sentences[i]
		senti, polar = 





