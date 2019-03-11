import pandas as pd
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams
import string
from progress.bar import Bar

filepath = '../data/filtered_train_data_all.csv'
# filepath = 'toy_set.csv'

df = pd.read_csv(filepath, index_col=0)

stopwords = set(stopwords.words('english'))
punctuation = string.punctuation

def sentence_sentiment(s):
	testimonial = TextBlob(s)
	# The polarity score is a float within the range [-1.0, 1.0].
	# The subjectivity is a float within the range [0.0, 1.0] where 0.0 is very objective and 1.0 is very subjective.
	return testimonial.sentiment.polarity, testimonial.sentiment.subjectivity

def stop_word_removal(s):
	tokens = word_tokenize(s)
	filtered = [w for w in tokens if w not in stopwords and w not in punctuation and not w.isnumeric()]
	return filtered

def sentence_processing(df):
	""" This function performs two tasks: 
	- Sentiment extraction and write the results into a csv file
	- Write filtered words into two text files, one for labelled data, one for unlabelled data
	"""
	pos_df, neg_df = df[df['target']==1], df[df['target']==0]
	sentences, targets = df['question_text'].values, df['target'].values
	sentiment_dict = {'sentiment':[], 'polarity':[], 'target':[]}
	pos_tokens = []
	neg_tokens = []

	bar = Bar("Collecting data over sentences", max=len(sentences))
	for i in range(len(sentences)):
		sentence = sentences[i]
		target = targets[i]
		# sentiment, polarity = sentence_sentiment(sentence)
		# sentiment_dict['sentiment'].append(sentiment)
		# sentiment_dict['polarity'].append(polarity)
		# sentiment_dict['target'].append(target)

		wordnet_lemmatizer = WordNetLemmatizer()
		filtered = stop_word_removal(sentence)
		stemmed = [wordnet_lemmatizer.lemmatize(w.lower()) for w in filtered]

		if target:
			pos_tokens.extend(stemmed)
		else:
			neg_tokens.extend(stemmed)

		bar.next()
	bar.finish()

	sentiment_df = pd.DataFrame(sentiment_dict)

	return pos_tokens, neg_tokens, sentiment_df

def most_frequent_unigram(tokens, n):
	""" Return the most frequent n unigrams in tokens. 
	"""
	frequency_dist = nltk.FreqDist(tokens)
	most_common = frequency_dist.most_common(n)
	return most_common

def main():
	pos_tokens, neg_tokens, sentiment_df = sentence_processing(df)

	# sentiment_df.to_csv('sentiment.csv')

	# top 200 tokens
	pos_unigram, neg_unigram = most_frequent_unigram(pos_tokens, 200), most_frequent_unigram(neg_tokens, 200)

	with open('pos_unigram.txt', 'w') as file:
		file.write(str(pos_unigram))

	with open('neg_unigram.txt', 'w') as file:
		file.write(str(neg_unigram))

main()