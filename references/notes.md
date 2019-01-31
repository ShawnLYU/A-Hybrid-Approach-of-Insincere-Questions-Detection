## References Quick Note

#### [Automatic Sarcasm Detection](jinny/sarcasm_detection.pdf)
(review: source of original articles)

- Approaches to detect sarcasm
    + rule-based
        * phrases that bear sentiment
        * coexistence of negative and positive phrases in the same sentence
    + feature
        * **73:11 contains a list of features used for statistical classifier**
        * [Are Word Embedding-based Features Useful for Sarcasm Detection](jinny/word_embedding_sarcasm.pdf)
            - use of word embedding-based features to augment word embeddings (useful in sarcasm detection)
                + word embedding similarity between words
    + Learning/deep learning/CNN
        * see below for original articles

#### [Fracking Sarcasm using Neural Network](jinny/cnn_lstm_sarcasm.pdf)
- CNN + LSTM(RNN) + DNN

| neural network | pro | usage |
| ------- | :------- | ------- |
| CNN | **pros:**</br> convolutional filter to capture temporal text sequence</br>reduce frequency variation </br> deals with the situation where relevant features can come from any part of the text </br> **cons:** </br>fixed filter width: cannot resolve when text is longer than 5 | map input feature to composite feature to feed into LSTM |
| LSTM | easy to train </br> no vanishing/exploding gradients during backprop </br> long-distance memory + feedback cycles </br> no fixed context window size | semantic modeling | 
| fully-connected DNN | mapping features into a more separable space | provide better classification |

- other components of the network
    + input to embedding layer: get distributional vector
    + softmax on top of DNN
        * higher softmax class probability makes conflicts between human-annotation and model output negligible
    + minimize binary cross-entropy
    + parameter optimization: ADAM (lr=0.001)
    + dropout layer: hinders performance in this study because some relevant features are lost in the dropout layer

#### [A Deeper Look into Sarcastic Tweets Using Deep Convolutional Neural Networks](jinny/cnn_sarcasm_tweets.pdf)
- Separately trained 4 CNN models for sentiment, emotion, personality, and baseline feature (direct text)
    + pretrain sentiment, emotion, and personality models on their corresponding datasets
    + pretrained models then used to extract features from twitter dataset
- Contains a detailed CNN architecture
    + word embedding: 
        * *word2vec*
        * non-static representations: word vectors into parameters to be learned
            - reason1: unavailability of informal language in the library
            - reason2: update embeddings to incorporate sentiment shift (specific for sarcasm detection)
    + CNN-SVM
        * output of CNN fed into SVN for classification
- Combining networks

 ![alt text](cnn-combination.png)

#### [Deep Convolutional Neural Networks for Sentiment Analysis of Short Texts](jinny/deep_cnn_for_sentiment_analysis.pdf)
(original)

- Twitter dataset: noisy positive/negative labels
- Character-level embeddings
    + capture morphological and shape info
    + use a convolutional layer to apply operations to successive windows of characters; max over all windows is the glocal chracter-level feature vector of the word
- Word-level embeddings
    + syntactic and semantic info
    + hyperparameter: size of word-level embedding
    + unsupervised training using *word2vec*
- Sentence-level
    + use word-level feature vectors to construct sentence-level feature set
    + similar to character-level extraction method


CNN setup:
- Convolutional layer: produce local features around each word/character in sentences/words
- Training: minimize a negative likelihood using stochastic gradient descent 
- learning rate had the largest impact on performance


#### [Robust Sentiment Detection on Twitter from Biased and Noisy Data](jinny/sentiment_detection_noisy.pdf)
(method too old; can be used in literature review)
(original)

- 2 steps: Subjectivity detection + polarity detection
- features
    + metafeature
        * POS tagging (many adj?)
    + tweet syntax features
- dealing with noise
    + data cleaning (might not apply in our project)
    + use manually tagged test data
- learning algorithm: SVM

#### [On so-called rhetorical questions](jinny/rhetorical_questions)
- Linguistic background knowledge for rhetorical questions, might be useful in feature engineering 
    + Question and answer co-exist
    + formal indications of rhetorial/assertive questions:
        * intonation pattern (up and then down), special particles(adverbials), verbal mood
        * non-deontic modal verbs (modal verbs that do not express permission/obligations to do something)
        * negation (which resonable man would vote conservatives? - no reasonable man would vote conservatives)
        * comparison with an absolute value (could there be a safer place than prison)
        * proform(interrogative pronoun) + expression of excusive absoluteness (
        EEA) + descriptive term
            - EEA: other than, if not (who else burns a cheque if not an idiot?)

