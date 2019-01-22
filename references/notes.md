## References Quick Note

#### [Deep Convolutional Neural Networks for Sentiment Analysis of Short Texts](deep_cnn_for_sentiment_analysis.pdf)
(original)

- Twitter dataset: noisy positive/negative labels
- Character-level embeddings
    + character embedding for each word
- Unsupervised learning for word-level embeddings (tool word2vec)
    + syntactic, semantic
- Sentence-level: use word-level features to construct sentence-level feature set


CNN setup:
- Convolutional layer: produce local features around each word/character in sentences/words
- Training: minimize a negative likelihood
- learning rate had the largest impact on performance

#### [Automatic Sarcasm Detection](sarcasm_detection.pdf)
(review)

- Approaches to detect sarcasm

    + rule-based
        * phrases that bear sentiment
        * coexistence of negative and positive phrases in the same sentence
    + feature
    + Learning/deep learning/CNN

#### [Robust Sentiment Detection on Twitter from Biased and Noisy Data](sentiment_detection_noisy.pdf)
**Highlight!!**

(original)

- Subjectivity detection
- Subjective tweets categorization (negative, neutral, positive)
- features
    + metafeature
        * POS tagging (many adj?)
    + tweet syntax features

