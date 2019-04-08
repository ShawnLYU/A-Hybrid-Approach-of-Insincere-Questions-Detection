# A Hybrid Approach to Insincere Questions Detection

Detailed report could be found at _[A Hybrid Approach of Insincere Questions Detection](https://github.com/ShawnLYU/Quora-Insincere-Questions-Classification/blob/master/report/csc2511.pdf)_.

## Description
This is a NLP project to detect toxic content to improve online conversations supervised by [Prof. Rudzicz](http://www.cs.toronto.edu/~frank/) @ University of Toronto.


As Q\&A websites become more prevalent, it is critical to detect and reveal patterns of misleading or discriminative questions. In this project, we studied Quora corpus with the focus on questions with non-neutral tone, rhetorical questions, discriminative questions, and questions with sexual contents as shock values. We discovered that insincere questions share substantial similarities with sarcasm in terms of their linguistic nature, and sarcasm detection is a widely studied subtopic of sentiment analysis in the field of natural language processing. Therefore,  we transferred and further extended state-of-art sarcasm detection approaches to insincere question classification. In this paper, we proposed a baseline model which is composed of CNN-LSTM and a hybrid model argumented with corpus statistics. Both models utilized [ELMo](https://allennlp.org/elmo) as the embedding layer, which were fine-tuned within the framework provided by [Allennlp](https://allennlp.org/). To explore the scientific nature of the phenomenon, we conducted an in-depth corpus analysis on Quora questions in four aspects, namely punctuation marks, part-of-speech tags, named entities, and sentiment scores. The analysis revealed significant differences between sincere and insincere questions morphologically, syntactically, and semantically. Lastly, we performed evaluations using precision-recall metrics, which verified our hypothesis that the hybrid model would outperform the baseline model. 


<p align="center">
  <img src='https://github.com/ShawnLYU/Quora-Insincere-Questions-Classification/blob/master/report/graphs/nn_architecture.png'/>
</p>

## Getting Started

These repo will get you a copy of both our baseline model and hybrid model, through which you could have your own NLP analysis. Sample data is provided here for to get your hands dirty.


### Prerequisites

What things you need to install the software and how to install them

- Python 3.7.0
- [Allennlp](https://github.com/allenai/allennlp)
- Numpy, Sklearn, Pandas



### Installing

A step by step series of examples that tell you how to get a development env running:

Fistly you need to download this repo and there are a few things you need to setup before running the model

- `/model/baseline_model/config.json`, line 10 & 11
  You need to setup the absolute path for the training data and validation data to `/dev_data/train_0125.csv` and `/dev_data/valid_0125.csv`

- `/model/hybrid_model/config.json`, line 10 & 11
  You need to setup the absolute path for the training data and validation data to `/dev_data/train_0125.csv` and `/dev_data/valid_0125.csv`

- `/model/hybrid_model/myutils.py`, line 4
  You need to setup the absolute path to `/dev_data_stats`


## Running the tests

To run the model, you can cd any of the baseline model or the hybrid model, and with command:

```
allennlp train config.json -s res --include-package packages
```

## Notes

This package would save the model after each epoch and all of the metrics during the training process.

## Contributing

This project exists thanks to all the people who contribute. 

[Shawn](https://github.com/ShawnLYU)    
[JinyueF](https://github.com/JinyueF)

## License

This project is licensed under the [MIT](LICENSE) License.

## Acknowledgments

Prof. Rudzicz provided inspirations and great helps.
