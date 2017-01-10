# google-word2vec-demo
A simple minimalistic demo of using google's pre-trained word2vec model

Recently, I have been trying to convert words into features so that they could be fed into models for many machine learning tasks
like text generation, text classification, sentiment analysis etc.

Instead of creating your own embedding from sentences you have, you could just use google's pre-trained embedding (.bin) and load them into word2vec model and do all you want to do... This is really useful and saves training time. Also, this pre-trained model consists of 3 billion words and each word is represented by a 300-D feature vector.

**To get started:**

* Install gensim (use PyPi or Anaconda) (I'm not going to tell how to do this :-)   )

* Download the  bin file (google's trained model) from [here](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit)

* Play around with embeddings using [fiddle.py] (https://github.com/rajcscw/google-word2vec-demo/blob/master/fiddle.py)


**What can you find in [fiddle.py] (https://github.com/rajcscw/google-word2vec-demo/blob/master/fiddle.py) ?**


* Imports gensim and loads pre-trained bin into word2vec model

```python
import gensim

# Load Google's pre-trained Word2Vec model
model = gensim.models.Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
```
* Shows the number of words which are in vocabulary
```python
# Checking vocabulary size
vocab = model.vocab.keys()
print("Vocabulary size is: " + str(len(vocab)))
```
* Finding similarity between two words 
```python
# Similarity between words
print(model.similarity("car", "vehicle"))
```
* Converting strings to raw vectors (This is the most important and relevant for ML tasks)
```python
# Getting embedding ( raw vectors of words)
car = model['car']
print(car.shape)
```

* Converting raw vectors back to strings
```python
# Getting word back from embedding
print(model.most_similar(positive=[car], negative=[], topn=1))
```

Note: You might encounter memory error or loading of model and most_similar() might take very long time which you have to overcome by using high end machines or diving deep into gensim API for optimizations.

References:
* [http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/)
* [inspect_word2vec] (https://github.com/chrisjmccormick/inspect_word2vec)
