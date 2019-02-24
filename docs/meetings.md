# 1<sup>st</sup> meeting, Jan 21 
### TODO
- At least 7 papers each, due: Jan 31
- Brainstorm about proposal questions(e.g. project title, challenges, and etc.)
- Start writing proposal on Jan 28
### Next meeting
- Jan 28

# 2<sup>nd</sup> meeting, Jan 28
### TODO
- Figure out procedures of this project
- Explore cutting edge techniques/methods of sentiment analysis
### Next meeting
- Jan 29

# 3<sup>rd</sup> meeting with Frank, Feb 4
- Tell our project story
- Embeddings
	- Explain word embeddings (Word2Vec, GloVe, and Bert)
	- feature based extraction (tokenization, PoS-tagging, number of adjectives)
- Explain downstream NN tasks
	- CNN-LSTM model
	- Merging word-embeddings with statistical features using DNN or SVM
- Questions
	- What if we train our own embeddings? Advantages would be: typos could be properly processed hopefully(they may appear to be similar to the correct-spelt ones);
	- Which part of our project would be the proper one for us to make innovative improvements?

### TODO
- Try ELMo
- Word embeddings could also count as corpus stats, when texts are embedded, (visualized) and analyzed
- Focusing on comparing different techniques/models, and providing explanations

Notes from Frank
- Try different word embeddings including Word2Vec, GloVe, ELMo, and Bert.
- Compare the word embeddings and come up with our own hypothesis (with necessary visualisations).
- Try PoS-tagging and feature extractions targeting at corpus stats.
- In terms of classifiers, we would try CNN-LSTM and Random Forest, SVM (provided by Sklearn). And compare the results.
- Dig deeper on relations between sarcasm and insincerity, and come up with our own hypothesis.

# 4<sup>th</sup> Feb 12
1. Corpus Stats - **Jinny**
	1. Tagging + labelling
	2. Tagging + non-labelling
	3. Non-tagging + labelling
	4. Non-tagging + labelling
	5. Non-word information analysis
		- punctuations
		- misspelled words? 
2. Word Embeddings - **Shawn**
	1. ELMo
	2. GloVe
	3. Word2Vec
	4. Bert
3. Report - **Shawn, Jinny**

# 4<sup>th</sup> Feb 18
1. Corpus Stats - **Jinny**       
	1. New graphs     
		- visualization improvements (better design; more complex graphs; more information)
		- ref: AntV
	2. Save stats as features
	3. Ask Frank for suggestions (a. b.)
		1. The main task will be to identify statistics that might affect classification. Mainly, this will involve looking at mismatches in labels (i.e., unbalanced data), and in an association between whatever features you extract and the classes. E.g., if you extract 100 features, do a significance test on each features between the labels, to see if one class 'behaves' very differently or one class or another.
2. Word embeddings - **Shawn**
	1. ELMo
	2. BERT
3. Report - **Shawn & Jinny**
	1. Look for more references about sentiment analysis
	
# 5<sup>th</sup> Feb 24
1. Corpus Stats
	1. Clean double quotes - **Jinny**
	2. Modify labels of samples involved with math symbols - **Shawn**
	3. Corpus stats visualization
2. Word embeddings
	1. Applied pre-trained ELMo - **Shawn**
	2. Embeddings visualization - **Shawn**
3. Machine learning (with corpus stats)
	1. SVM - **Shawn**
	2. LR/RF - **Jinny**
4. Neural Nets (two nets with more structures)
	1. Shawn
	2. Jinny
