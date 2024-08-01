# Literature Review

Approaches or solutions that have been tried before on similar projects.

**Summary of Each Work**:

- **Source 1**: Toborek, V., Busch, M., Boßert, M., Bauckhage, C., & Welke, P. (2023). A new aligned simple German corpus. *Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, 11393–11412.
  - [**Link**](https://doi.org/10.48550/arXiv.2209.01106)
  - **Objective**: The paper presents a content-matched corpus of plain and regular German sentences. Additionally, individual sentences have been algorithmically aligned.
  - **Methods**: Parallel versions of plain and regular German articles were scraped from eight different web sources. Automatical sentence alignment was done using two versions of the Most Similar Text algorithm.
  - **Outcomes**: The final dataset consisted of 708 articles in plain German and 712 articles in regular German, with approximately 30,000 sentences for each plain and regular German. Of these, 10,304 matched plain German and regular German sentence-pairs were identified, with F1 scores of .28, which is an improvement over previous alignment methods.
  - **Relation to the Project**: The corpus of plain and regular German sentences can be used to train deep-learning models to distinguish between these two forms of language. Sentence alignment is not required for this text-classification task, but would be helpful for more advanced tasks, such as machine translation.
- **Source 2**: Text classification with TensorFlow Hub: Movie reviews
  - [**Link**](https://www.tensorflow.org/tutorials/keras/text_classification_with_hub)
  - **Objective**: This tutorial provides an example of a binary text-classification task (sentiment analysis for movie reviews) using a pre-trained embedding model.
  - **Methods**: A Neural-Net Language Model trained by Google is used as the pre-trained embedding model. Apart from the pre-trained model, the final model is a simple dense neural network with one hidden layer consisting of 16 neurons. For training, 15,000 reviews are used, with the entire IMDb corpus consisting of 50,000 reviews.
  - **Outcomes**: The model achieves an accuracy of about 87%.
  - **Relation to the Project**: The method of text classification using pre-trained embeddings showcased in this tutorial can be adapted to the project task by using a German embedding model instead of an English one.
- **Source 3**: Chan, B., Schweter, S., & Möller, T. (2020). German's next language model. *arXiv*. Advance onlince publication.
  - [**Link**](https://doi.org/10.48550/arXiv.2010.10906)
  - **Objective**: This paper introduces German versions of Bidirectional Encoder Representations from Transformers (BERT) and Efficiently Learning an Encoder that Classifies Token Replacements Accurately (ELECTRA).
  - **Methods**: The models are trained on a total of 163.4 GB of monolingual German text data scraped from the internet in general and Wikipedia in particular as well as texts from various domains including movie subtitles, parliament speeches, books, and court decisions.
  - **Outcomes**: The models exceed previous state-of-the-art performance on text-classification and named-entity-recognition tasks.
  - **Relation to the Project**: Fine tuning German BERT or a similar Transformer-based model may yield a superior model compared to training a model from scratch or using pre-trained embeddings.