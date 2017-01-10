import gensim

# Load Google's pre-trained Word2Vec model
model = gensim.models.Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

# Checking vocabulary size
vocab = model.vocab.keys()
print("Vocabulary size is: " + str(len(vocab)))

# Similarity between words
print(model.similarity("car", "vehicle"))

# Getting embedding ( raw vectors of words)
car = model['car']
print(car.shape)

# Getting word back from embedding
print(model.most_similar(positive=[car], negative=[], topn=1))