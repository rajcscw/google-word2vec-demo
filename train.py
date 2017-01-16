import codecs
from nltk import word_tokenize, sent_tokenize
from nltk.tokenize import WordPunctTokenizer
import gensim

filename = "wonderland.txt"
raw_text = codecs.open(filename, "r", "utf-8-sig").read()

# Remove newline and carraige returns
text = raw_text.replace("\r\n", ' ')

# Sentences
sentences = sent_tokenize(text)

# Iterate over each sentence and form list of lists (list of ( list of words))
training_list = []
word_tokenizer = WordPunctTokenizer()
for sentence in sentences:
    words = word_tokenize(sentence)
    training_list.append(words)

# train word2vec model
model = gensim.models.Word2Vec(training_list, size=5, window=10, min_count=1, workers=4)
model.save("alice_model")
model = gensim.models.Word2Vec.load("alice_model")

# Checking vocabulary size
vocab = model.vocab.keys()
print("Vocabulary size is: " + str(len(vocab)))
print(vocab)

# Similarity between words
print(model.similarity("Alice", "rabbit"))

# Getting embedding ( raw vectors of words)
rabbit = model['rabbit']
print(rabbit.shape)



