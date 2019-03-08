import pandas as pd 
from matplotlib import pyplot as plt

ent_df = pd.read_csv('ent.csv', index_col=0)
punc_df = pd.read_csv('punctuation.csv', index_col=0)
pos_df = pd.read_csv('pos_tag.csv', index_col=0)

def split_target(df):
	return df[df['target'] == 1], df[df['target'] == 0]

def create_count_df_dict(pos_df, neg_df, normalize=True):
	result = {}
	for label in pos_df.columns:
		if label != 'target':
			count_df =  pd.DataFrame(data={
				'positive': pd.Series(pos_df[label]).value_counts(normalize=normalize), 
				'negative': pd.Series(neg_df[label]).value_counts(normalize=normalize)
				})
			result[label] = count_df
	return result

def create_value_df_dict(pos_df, neg_df, normalize=True):
	result = {}
	for label in pos_df.columns:
		if label != 'target':
			values={
				'positive': list(pos_df[label].values), 
				'negative': list(neg_df[label].values)}
			result[label] = values
	return result

def corpus_level_plot(df_dict, kind):
	nrow = (len(df_dict.keys())+1) // 2
	ncol = 2
	figure, axes = plt.subplots(nrow, ncol)
	x = 0
	y = 0
	
	for key in df_dict:
		if kind == 'bar':
			curr_df = df_dict[key]
			title = kind + ' plot for ' + key
			curr_df.plot(kind=kind, title=title, legend=True, ax=axes[x,y], xticks=range(0, int(curr_df.max().max())))

		# update ax_pos
		if y == 0:
			y = 1
		else:
			x += 1
			y = 0

	return figure, axes

def plot_test():
	df_list = [ent_df, punc_df, pos_df]
	labels_list = [df.columns for df in df_list]
	splitted_dfs = [split_target(df) for df in df_list]

	normalized_count_dicts = [create_count_df_dict(dfs[0], dfs[1]) for dfs in splitted_dfs]
	value_dicts = [create_value_df_dict(dfs[0], dfs[1]) for dfs in splitted_dfs]
	print(value_dicts[0].keys())

	bar_figures = [corpus_level_plot(df_dict, 'bar') for df_dict in normalized_count_dicts]
	
	pos_values = value_dicts[2]

	box_figures = []
	for key in pos_values:
		figure, ax = plt.subplots()
		labels, data = pos_values[key].keys(), pos_values[key].values()
		ax.boxplot(data, labels=labels)
		ax.set_title(key)
		box_figures.append(figure)

	def count_value(df):
		result = {}
		for key in df.columns:
			if key != 'all_punctuation' and key != 'target':
				result[key] = sum(df[key].values)
		return result

	pos_punc_count = count_value(splitted_dfs[1][0])
	neg_punc_count = count_value(splitted_dfs[1][1])

	colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99', '#adadeb']

	plt.figure('puncpiecharts')
	plt.subplot(121, title='Positive Samples')
	labels = list(pos_punc_count.keys())
	counts = list(pos_punc_count.values())
	plt.pie(counts, colors=colors, labels=labels, autopct='%1.1f%%')

	plt.subplot(122, title='Negative Samples')
	labels = list(neg_punc_count.keys())
	counts = list(neg_punc_count.values())
	plt.pie(counts, colors=colors, labels=labels, autopct='%1.1f%%')


plot_test()
plt.show()
