# A Hybrid Approach to Insincere Questions Detection

Detailed report could be found at _[A Hybrid Approach to Insincere Questions Detection](https://github.com/ShawnLYU/Quora-Insincere-Questions-Classification/blob/master/report/csc2511.pdf)_.

## Description
This is a NLP project to detect toxic content to improve online conversations supervised by [Prof. Rudzicz](http://www.cs.toronto.edu/~frank/) @ University of Toronto.

As Q\&A sites become more prevalent, it is critical to detect and reveal patterns of misleading or discriminative questions. In this project, we studied Quora with the focus on questions with non-neutral tone, rhetorical questions, discriminative questions, and questions with sexual contents as shock values. In natural language processing, a lot of research has been conducted over sarcasm, and insincere questions share substantial similarities with it from linguistics interpretations. Therefore,  we transferred and further extend state-of-art approaches of sarcasm detections. In this paper, we proposed a baseline model which is composed of CNN-LSTM and a hybrid model argumented with corpus statistics. Both models utilise [ELMo](https://allennlp.org/elmo) as the embedding layer and fine-tuned it within the framework provided by [Allennlp](https://allennlp.org/). Second, we conducted an in-depth corpus analysis over Quora questions, such as punctuation scores, part-of-speech tagging, named entities, and sentiment scores. Third, we performed evaluations with precision-recall metrics and compare the results with the intended hypothesis. Finally, our hybrid model claimed to outperform the baseline model.

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

This project is licensed under the [MIT](LICENSE) License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

Prof. Rudzicz provided inspirations and great helps.
