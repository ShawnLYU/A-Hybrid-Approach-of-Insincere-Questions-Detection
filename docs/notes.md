# The story
1. Sarcasm classification is a relatively well-studied topics in sentiment analysis. As insincerity is related with sarcasm in many ways(proved), models/methods of sarcasm classification may provide us with inspirations looking into insincere detection. Therefore we plan to review the latest models/methods from sarcasm detection and customize them into this project.    
2. To embed texts into numerical representations, apart from work embeddings that are widely applied in NLP tasks, we would also try feature extraction with corpus statistics, such as the number of adjectives. Then we would compare the performances and try to combine these two ways. We expect that the combination would produce better results.


# Data
- Size
	1. training data: 1,306,122 * 3
	2. test data: 56,370 * 2
- Embeddings
	1. glove: 5.3G
	2. google: 3.4G
	3. para: 4.3G


# Challenges
- Spell error
- Short text V.S. long text
- Traditional methods of preprocessing (removing stopwords, stemming, and etc.) may cause information lost which would have been utilized by NN
- Word representations
- Tokenize the string    
    How to split regarding the punctuations?

# Definition of insincere questions
> - Has a non-neutral tone

Sentiment analysis to detect non-neutral tone

> - Has an exaggerated tone to underscore a point about a group of people
>       Is rhetorical and meant to imply a statement about a group of people
>       Is disparaging or inflammatory
>       Suggests a discriminatory idea against a protected class of people, or seeks confirmation of a stereotype
>       Makes disparaging attacks/insults against a specific person or group of people 

Use linguistic knowledge to design related features?

> - Based on an outlandish premise about a group of people
>       Disparages against a characteristic that is not fixable and not measurable

Word embedding similarity to a set of words that refer to groups of people?

> - Isn't grounded in reality
>       Based on false information, or contains absurd assumptions

No related feature? Rely on distributed representation?

> - Uses sexual content (incest, bestiality, pedophilia) for shock value, and not to seek genuine answers

Explicit rules for sexual content detection or rely on distributed representation? 

# TODO before next checkpoint (Mar5<sup>th</sup>)
1. Programming component and empirical methods
2. Reports about the design of the program and the setup environment.

# Newest updates from Shawn (Mar 31<sup>rd</sup>)
- split train(835828, 9), valid(209011, 9), test(261257, 4)
- add 'tag' to train, valid, test, so that train['tag']='train', valid['tag']='valid', test['tag']='test'
- Remove samples with more than 16 sentence length in train  
train:  
sen percentage  
10 0.2608012653321018  
11 0.34903831888857517  
12 0.432684715037065  
13 0.506616193762353  
14 0.5710361461927573  
15 0.6260845532812971  
16 0.6731145642404897  
17 0.7133094368697867  
18 0.7477363763836579  
19 0.7772711610522739  

valid:  
sen percentage  
10 0.2629191765026721  
11 0.350900191855931  
12 0.434809651166685  
13 0.5091215294888786  
14 0.5733956586017004  
15 0.6287946567405543  
16 0.6757156321916071  
17 0.7152733588184353  
18 0.7493672581825837  
19 0.778533187248518  
  
train(596204, 9), valid(149500, 9), test(261257, 4)

# Final Report TODO list from Jinny (Apr 2nd)
1. Description of the model
	- details about the architecture
	- hyperparameters choices and reasoning (may need a chart)
	- a more detailed architecture graph to show the dimensions?

2. Corpus stats analysis
	- visualization
		- current plots
		- PCA for feature vector visualization
		- word vector clustering
	- textual analysis
		- compare and contrast features of positive and negative data
		- a look into the expressive nature of insincere questions

3. NN model analysis
	- Baseline model
		- performance report
		- result analysis: 
			- why it works 
			- brief review of the relation to sarcasm detection
			- limitaion of the current implementation
			- possible improvements
	- Adding extracted features
		- performance report
		- analysis on the effects of these features
			- reasoning in general
			- can we possibly get the features that contributed the most? 
		- limitation
		- possible improvements
	- a brief conclusion
		- what we have contributed:
			- transferring approaches from sarcasm detection to insincerity detection
			- the integration of two different approaches

		

# Experiments
- basic setups: __model1__..................... __model_with_stats__
- with no dropouts: __model1_nodropout__..................... __model_with_stats_nodropout__
- with smaller LSTM hidden size: __model1_3_smaller_hidden_size__................ __model_with_stats_3_smaller_hidden_size__
- with different CNN-kernel size:









