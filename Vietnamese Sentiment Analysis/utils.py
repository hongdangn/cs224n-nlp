from underthesea import word_tokenize
import torch
import torchtext.vocab as vocab
import pandas as pd

word_embeddings = vocab.Vectors('dataset/vi_word2vec.txt', unk_init = torch.Tensor.normal_)
df = pd.read_csv('dataset/VI_IMDB.csv')

class Vocabulary():
  def __init__(self):
    self.word2id = dict()
    self.word2id['<pad>'] = 0
    self.word2id['<unk>'] = 1
    self.id2word = {id : word for word, id in self.word2id.items()}

  def id2word(self, id):
    return self.id2word[id]

  def add(self, word):
    word_index = len(self.word2id)
    self.word2id[word] = word_index
    self.id2word[word_index] = word

  def corpus_to_tensor(self, corpus):
    tokens = word_tokenize(corpus)
    indicies = list()
    for word in tokens:
      word = word.replace(" ", "_")
      indice_word = self.word2id[word] if word in self.word2id else self.word2id['<unk>']
      indicies.append(indice_word)
    return indicies

  def tensor_to_corpus(self, tensor):
    corpus = list()
    for index in tensor:
      corpus.append(self.id2word[index])
    return corpus

Vocab = Vocabulary()

for word in word_embeddings.stoi:
  Vocab.add(word)