import pandas as pd 
from matplotlib import pyplot as plt
from corpus_stats import ks_test

df = pd.read_csv('sentiment.csv', index_col=0)

pos_df, neg_df = df[df['target']==1], df[df['target']==0]

for label in list(df):
	if label != 'target':
		ks_test(pos_df[label].values, neg_df[label].values, label)

