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
