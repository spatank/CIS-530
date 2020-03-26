# from nltk.corpus import conll2002
import nltk 
nltk.download('conll2002')
from nltk.corpus import conll2002
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Perceptron, RidgeClassifier, PassiveAggressiveClassifier
from sklearn.metrics import precision_recall_fscore_support
import pickle

# Assignment 7: NER
# Rebecca Iglesias-Flores and Shubhankar Patankar
# This is just to help you get going. Feel free to
# add to or modify any part of it.

# Assignment 7: NER
# This is just to help you get going. Feel free to
# add to or modify any part of it.


def hasDot(word):
  return int('.' in word)

def hasApost(word):
  return int('\'' in word)

def hasHyph(word):
  return int('-' in word)

def hasNC(pos):
  return int('NC' in pos)

def hasAQ(pos):
  return int('AQ' in pos)

def isCap(word):
    ## if first letter of the word is capitalized
  return int(word[0].isupper()) 

def hasCap(word):
    ## if any letter of the word is capitalized or not
  return int(word.islower())

accents = 'ÂÃÄÀÁÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÛÜÝÞßàáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿ'

def hasAcc(word):
  for char in word:
    if char in accents:
      return int(True)
  return int(False)

def hasDig(word):
  ## assigns a 1 if there's a digit in the word
  return int(word.isalpha())

def prefix(word):
  return word[:4]

def suffix(word):
  return word[-4:]

def wordshape(word):
    import re
    t1 = re.sub('[A-Z]', 'X', word)
    t2 = re.sub('[a-z]', 'x', t1)
    return re.sub('[0-9]', 'd', t2)

def getfeats(word, pos_tag, o):
  """ This takes the word in question and
  the offset with respect to the instance
  word """
  o = str(o)
  features = [
              (o + 'word', word),
              (o + 'shape', wordshape(word)),
              # TODO: add more features here.
              # (o + 'hasDot', hasDot(word)),
              (o + 'hasApost', hasApost(word)),
              (o + 'hasHyph', hasHyph(word)),
              (o + 'hasNC', hasNN(pos_tag)),
              (o + 'hasAQ', hasNN(pos_tag)),
              (o + 'isCap', isCap(word)),
              (o + 'hasCap', hasCap(word)),
              (o + 'hasAcc', hasAcc(word)),
              # (o + 'hasDig', hasDig(word)),
              (o + 'prefix', prefix(word)),
              (o + 'suffix', suffix(word))
              ]
  return features
    
def word2features(sent, i):
  """ The function generates all features
  for the word at position i in the
  sentence."""
  features = []
  # the window around the token
  for o in [-3,-2,-1,0,1,2,3]:
    if i+o >= 0 and i+o < len(sent):
      word = sent[i+o][0]
      pos_tag = sent[i+o][1]
      featlist = getfeats(word, pos_tag, o)
      features.extend(featlist)
  
  return dict(features)

if __name__ == "__main__":

  train_sents = list(conll2002.iob_sents('esp.train'))
  train_feats = []
  train_labels = []
  for sent in train_sents:
    for i in range(len(sent)):
      feats = word2features(sent,i)
      train_feats.append(feats)
      train_labels.append(sent[i][-1])

  vectorizer = DictVectorizer()
  X_train = vectorizer.fit_transform(train_feats)

  # model = Perceptron(verbose = 1, max_iter = 2000)
  # model = RidgeClassifier()
  model = PassiveAggressiveClassifier(verbose = 1, loss = 'squared_hinge')
  model.fit(X_train, train_labels)
  pickle.dump(model, open('drive/My Drive/CIS-530/Homework 7/Results/model', 'wb'))

  # TRAINING SET
  y_pred_train = model.predict(X_train)  
  j = 0
  print("Writing to train_results.txt")
  # format is: word gold pred
  outfile_path = 'drive/My Drive/CIS-530/Homework 7/Results/train_results.txt' 
  with open(outfile_path, "w") as out:
    for sent in train_sents: 
      for i in range(len(sent)):
        word = sent[i][0]
        gold = sent[i][-1]
        pred = y_pred_train[j]
        j += 1
        out.write("{}\t{}\t{}\n".format(word, gold, pred))
    out.write("\n")

  # DEVELOPMENT SET
  dev_sents = list(conll2002.iob_sents('esp.testa'))
  dev_feats = []
  dev_labels = []
  for sent in dev_sents:
    for i in range(len(sent)):
      feats = word2features(sent,i)
      dev_feats.append(feats)
      dev_labels.append(sent[i][-1])
  X_dev = vectorizer.transform(dev_feats)
  y_pred_dev = model.predict(X_dev)
  j = 0
  print("Writing to dev_results.txt")
  # format is: word gold pred
  outfile_path = 'drive/My Drive/CIS-530/Homework 7/Results/dev_results.txt'
  with open(outfile_path, "w") as out:
    for sent in dev_sents: 
      for i in range(len(sent)):
        word = sent[i][0]
        gold = sent[i][-1]
        pred = y_pred_dev[j]
        j += 1
        out.write("{}\t{}\t{}\n".format(word,gold,pred))
    out.write("\n")

  # TEST SET
  test_sents = list(conll2002.iob_sents('esp.testb'))
  test_feats = []
  test_labels = []
  for sent in test_sents:
    for i in range(len(sent)):
      feats = word2features(sent,i)
      test_feats.append(feats)
      test_labels.append(sent[i][-1])
  X_test = vectorizer.transform(test_feats)
  y_pred_test = model.predict(X_test)
  j = 0
  print("Writing to test_results.txt")
  # format is: word gold pred
  outfile_path = 'drive/My Drive/CIS-530/Homework 7/Results/test_results.txt'
  with open(outfile_path, "w") as out:
    for sent in test_sents: 
      for i in range(len(sent)):
        word = sent[i][0]
        gold = sent[i][-1]
        pred = y_pred_test[j]
        j += 1
        out.write("{}\t{}\t{}\n".format(word,gold,pred))
    out.write("\n")

  print("Now run: python conlleval.py test_results.txt")
