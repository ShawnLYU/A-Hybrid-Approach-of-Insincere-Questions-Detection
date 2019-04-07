# Quora-Insincere-Questions-Classification
This is a NLP project to detect toxic content to improve online conversations supervised by [Prof. Rudzicz](http://www.cs.toronto.edu/~frank/) @ University of Toronto.

As Q\&A sites become more prevalent, it is critical to detect and reveal patterns of misleading or discriminative questions. In this project, we studied Quora with the focus on questions with non-neutral tone, rhetorical questions, discriminative questions, and questions with sexual contents as shock values. In natural language processing, a lot of research has been conducted over sarcasm, and insincere questions share substantial similarities with it from linguistics interpretations. Therefore,  we transferred and further extend state-of-art approaches of sarcasm detections. In this paper, we proposed a baseline model which is composed of CNN-LSTM and a hybrid model argumented with corpus statistics. Both models utilise [ELMo](https://allennlp.org/elmo) as the embedding layer and fine-tuned it within the framework provided by [Allennlp](https://allennlp.org/). Second, we conducted an in-depth corpus analysis over Quora questions, such as punctuation scores, part-of-speech tagging, named entities, and sentiment scores. Third, we performed evaluations with precision-recall metrics and compare the results with the intended hypothesis. Finally, our hybrid model claimed to outperform the baseline model.



## Contributors
This project exists thanks to all the people who contribute. 

[Shawn](https://github.com/ShawnLYU)    
[JinyueF](https://github.com/JinyueF)



## License

[MIT](LICENSE)
Copyright (c) 2019 Shawn, JinyueF
