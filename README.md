# google-word2vec-demo
A demo of using google's pre-trained word2vec model

Recently, I have been trying to convert words into features so that it could be fed into models for many machine learning tasks
like text generation, text classification, sentiment analysis etc.

Instead of creating from own embedding, one could use google's pre-trained embedding (.bin) and load them into word2vec model and do all you have to do... This is really useful and saves all your training time. Also, this pre-trained model consists of 3 billion words and each word is represented by a 300-D feature vector.

To get started:

1) Install gensim

2) Download the  bin file (google's trained model)

3) Play around with embeddings using (fiddle.py)


What can you find in fiddle.py:
