#############################################################
## ASSIGNMENT 2 CODE SKELETON
## RELEASED: 1/29/2019
## DUE: 2/5/2019
## DESCRIPTION: In this assignment, you will explore the
## text classification problem of identifying complex words.
## We have provided the following skeleton for your code,
## with several helper functions, and all the required
## functions you need to write.
#############################################################

from collections import defaultdict
import gzip
import numpy as np
import re
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import nltk
nltk.download('wordnet')
nltk.download('punkt')
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize

#### 1. Evaluation Metrics ####

## Input: y_pred, a list of length n with the predicted labels,
## y_true, a list of length n with the true labels

## Calculates the precision of the predicted labels
def get_precision(y_pred, y_true):
    # true positive: system says positive when truly positive
    true_positives = len([pred for idx, pred in enumerate(y_pred) if pred == 1 if y_true[idx] == 1])
    # false positive = system says positive when truly negative
    false_positives = len([pred for idx, pred in enumerate(y_pred) if pred == 1 if y_true[idx] == 0])
    precision = true_positives / (true_positives + false_positives)
    return precision
    
## Calculates the recall of the predicted labels
def get_recall(y_pred, y_true):
    # true positive: system says positive when truly positive
    true_positives = len([pred for idx, pred in enumerate(y_pred) if pred == 1 if y_true[idx] == 1])
    # false negative = system says negative when truly positive
    false_negatives = len([pred for idx, pred in enumerate(y_pred) if pred == 0 if y_true[idx] == 1])
    recall = true_positives / (true_positives + false_negatives)
    return recall

## Calculates the f-score of the predicted labels
def get_fscore(y_pred, y_true):
    precision = get_precision(y_pred, y_true)
    recall = get_recall(y_pred, y_true)
    fscore = (2 * precision * recall) / (precision + recall)
    return fscore

#### 2. Complex Word Identification ####

## Loads in the words and labels of one of the datasets
def load_file(data_file):
    words = []
    labels = []   
    with open(data_file, 'rt', encoding = "utf8") as f:
        i = 0
        for line in f:
            if i > 0: # skips over column headers
                line_split = line[:-1].split("\t")
                words.append(line_split[0].lower())
                labels.append(int(line_split[1]))
            i += 1
    return words, labels

### 2.1: A very simple baseline

## Makes feature matrix for all complex
def all_complex_feature(words):
    return [1 for word in words]

## Labels every word complex
def all_complex(data_file):
    words, y_true = load_file(data_file)
    y_pred = all_complex_feature(words)
    precision = get_precision(y_pred, y_true)
    recall = get_recall(y_pred, y_true)
    fscore = get_fscore(y_pred, y_true)
    performance = [precision, recall, fscore]
    return performance


### 2.2: Word length thresholding

## Makes feature matrix for word_length_threshold
def length_threshold_feature(words, threshold):
    feature = []
    for word in words:
        if len(word) >= threshold:
            feature.append(1)
        else:
            feature.append(0)
    return feature

## Finds the best length threshold by f-score, and uses this threshold to
## classify the training and development set
def word_length_threshold(training_file, development_file):
    # training set data
    words_train_file, labels_train_file = load_file(training_file)
    precisions = []
    recalls = []
    fscores = []
    thresholds = range(10)
    for threshold in thresholds:
        y_pred_train = length_threshold_feature(words_train_file, threshold)
        precisions.append(get_precision(y_pred_train, labels_train_file))
        recalls.append(get_recall(y_pred_train, labels_train_file))
        fscores.append(get_fscore(y_pred_train, labels_train_file))
    threshold_choice = np.argmax(fscores)
    # re-train with chosen hyperparameter value
    y_pred_train = length_threshold_feature(words_train_file, threshold_choice)
    tprecision = get_precision(y_pred_train, labels_train_file)
    trecall = get_recall(y_pred_train, labels_train_file)
    tfscore = get_fscore(y_pred_train, labels_train_file)
    training_performance = [tprecision, trecall, tfscore]
    # development set data
    words_dev_file, labels_dev_file = load_file(development_file)
    y_pred_dev = length_threshold_feature(words_dev_file, threshold_choice)
    dprecision = get_precision(y_pred_dev, labels_train_file)
    drecall = get_recall(y_pred_dev, labels_train_file)
    dfscore = get_fscore(y_pred_dev, labels_train_file)
    development_performance = [dprecision, drecall, dfscore]
    return training_performance, development_performance

### 2.3: Word frequency thresholding

## Loads Google NGram counts
def load_ngram_counts(ngram_counts_file): 
   counts = defaultdict(int) 
   with gzip.open(ngram_counts_file, 'rt') as f: 
       for line in f:
           token, count = line.strip().split('\t') 
           if token[0].islower(): 
               counts[token] = int(count) 
   return counts

# Finds the best frequency threshold by f-score, and uses this threshold to
## classify the training and development set

## Make feature matrix for word_frequency_threshold

def frequency_threshold_feature(words, threshold, counts):
    feature = []
    for word in words:
        if counts[word] < threshold: # infrequent words are complex
            feature.append(1)
        else:
            feature.append(0)
    return feature

def word_frequency_threshold(training_file, development_file, counts):
    # training set data
    words_train_file, labels_train_file = load_file(training_file)
    precisions = []
    recalls = []
    fscores = []
    thresholds = list(range(1000000, 70000000, 100000))
    for threshold in thresholds:
        y_pred_train = frequency_threshold_feature(words_train_file, threshold, counts)
        precisions.append(get_precision(y_pred_train, labels_train_file))
        recalls.append(get_recall(y_pred_train, labels_train_file))
        fscores.append(get_fscore(y_pred_train, labels_train_file))
    threshold_choice = thresholds[np.argmax(fscores)]
    # re-train with chosen hyperparameter value
    y_pred_train = frequency_threshold_feature(words_train_file, threshold_choice)
    tprecision = get_precision(y_pred_train, labels_train_file)
    trecall = get_recall(y_pred_train, labels_train_file)
    tfscore = get_fscore(y_pred_train, labels_train_file)
    training_performance = [tprecision, trecall, tfscore]
    # development set data
    words_dev_file, labels_dev_file = load_file(development_file)
    y_pred_dev = frequency_threshold_feature(words_dev_file, threshold_choice)
    dprecision = get_precision(y_pred_dev, labels_train_file)
    drecall = get_recall(y_pred_dev, labels_train_file)
    dfscore = get_fscore(y_pred_dev, labels_train_file)
    development_performance = [dprecision, drecall, dfscore]
    return training_performance, development_performance

### 2.4: Naive Bayes
        
## Trains a Naive Bayes classifier using length and frequency features
def naive_bayes(training_file, development_file, counts):
    
    words_train_file, labels_train_file = load_file(training_file)
    precisions = []
    recalls = []
    fscores = []
    thresholds = range(10)
    for threshold in thresholds:
        y_pred_train = length_threshold_feature(words_train_file, threshold)
        precisions.append(get_precision(y_pred_train, labels_train_file))
        recalls.append(get_recall(y_pred_train, labels_train_file))
        fscores.append(get_fscore(y_pred_train, labels_train_file))
    threshold_choice_length = np.argmax(fscores)
    train_length_feat = length_threshold_feature(words_train_file, threshold_choice_length)
    
    precisions = []
    recalls = []
    fscores = []
    thresholds = list(range(1000000, 70000000, 100000))
    for threshold in thresholds:
        y_pred_train = frequency_threshold_feature(words_train_file, threshold, counts)
        precisions.append(get_precision(y_pred_train, labels_train_file))
        recalls.append(get_recall(y_pred_train, labels_train_file))
        fscores.append(get_fscore(y_pred_train, labels_train_file))
    threshold_choice_freq = thresholds[np.argmax(fscores)]
    train_freq_feat = frequency_threshold_feature(words_train_file, threshold_choice_freq, counts)
    
    X_train = np.column_stack((train_length_feat, train_freq_feat))
    X_train = (X_train - X_train.mean(axis = 0)) / X_train.std(axis = 0) # normalize
    Y_train = np.asarray(labels_train_file)
    clf = GaussianNB()
    clf.fit(X_train, Y_train)
    Y_pred_train = clf.predict(X_train)
    tprecision = get_precision(Y_pred_train.tolist(), Y_train.tolist())
    trecall = get_recall(Y_pred_train.tolist(), Y_train.tolist())
    tfscore = get_fscore(Y_pred_train.tolist(), Y_train.tolist())
    training_performance = (tprecision, trecall, tfscore)
    
    words_dev_file, labels_dev_file = load_file(development_file)
    dev_length_feat = length_threshold_feature(words_dev_file, threshold_choice_length)
    dev_freq_feat = frequency_threshold_feature(words_dev_file, threshold_choice_freq, counts)
    X_dev = np.column_stack((dev_length_feat, dev_freq_feat))
    X_dev = (X_dev - X_train.mean(axis = 0)) / X_train.std(axis = 0) # normalize
    Y_dev = np.asarray(labels_dev_file)

    Y_pred_dev = clf.predict(X_dev)
    dprecision = get_precision(Y_pred_dev.tolist(), Y_dev.tolist())
    drecall = get_recall(Y_pred_dev.tolist(), Y_dev.tolist())
    dfscore = get_fscore(Y_pred_dev.tolist(), Y_dev.tolist())
    development_performance = (dprecision, drecall, dfscore)
    
    return development_performance

### 2.5: Logistic Regression

## Trains a Logistic Regression classifier using length and frequency features
def logistic_regression(training_file, development_file, counts):
    
    words_train_file, labels_train_file = load_file(training_file)
    num_words = len(words_train_file)
    num_features = 2
    X_train = np.zeros((num_words, num_features))
    Y_train = np.asarray(labels_train_file)
    for i, row in enumerate(X_train):
        row[0] = len(words_train_file[i])
        row[1] = counts[words_train_file[i]]
    mean1, mean2 = np.mean(X_train, axis = 0)
    std1, std2 = np.std(X_train, axis = 0)

    def normalize(X):
        return [((X[0]-mean1)/std1),((X[1]-mean2)/std2)]

    X_train = np.apply_along_axis(normalize, 1, X_train)
    clf = LogisticRegression()
    clf.fit(X_train, Y_train)
    Y_pred_train = clf.predict(X_train)
    tprecision = get_precision(Y_pred_train.tolist(), Y_train.tolist())
    trecall = get_recall(Y_pred_train.tolist(), Y_train.tolist())
    tfscore = get_fscore(Y_pred_train.tolist(), Y_train.tolist())
    training_performance = (tprecision, trecall, tfscore)


    words_dev_file, labels_dev_file = load_file(development_file)
    num_words = len(words_dev_file)
    num_features = 2
    X_dev = np.zeros((num_words, num_features))
    Y_dev = np.asarray(labels_dev_file)
    for i, row in enumerate(X_dev):
        row[0] = len(words_dev_file[i])
        row[1] = counts[words_dev_file[i]]

    X_dev = np.apply_along_axis(normalize, 1, X_dev)
    Y_pred_dev = clf.predict(X_dev)
    dprecision = get_precision(Y_pred_dev.tolist(), Y_dev.tolist())
    drecall = get_recall(Y_pred_dev.tolist(), Y_dev.tolist())
    dfscore = get_fscore(Y_pred_dev.tolist(), Y_dev.tolist())
    development_performance = (dprecision, drecall, dfscore)
    
    return development_performance

### 2.7: Build your own classifier

## Trains a classifier of your choosing, predicts labels for the test dataset
## and writes the predicted labels to the text file 'test_labels.txt',
## with ONE LABEL PER LINE

def count_syllables(word): 
    word = word.lower() 
    # exception_add are words that need extra syllables
    # exception_del are words that need less syllables
    exception_add = ['serious','crucial']
    exception_del = ['fortunately','unfortunately'] 
    co_one = ['cool','coach','coat','coal','count','coin','coarse','coup','coif','cook','coign','coiffe','coof','court']
    co_two = ['coapt','coed','coinci']
    pre_one = ['preach']
 
    syls = 0 # added syllable number
    disc = 0 # discarded syllable number
 
    #1) if letters < 3 : return 1
    if len(word) <= 3 :
        syls = 1
        return syls
 
    #2) if doesn't end with "ted" or "tes" or "ses" or "ied" or "ies", discard "es" and "ed" at the end.
    # if it has only 1 vowel or 1 set of consecutive vowels, discard. (like "speed", "fled" etc.) 
    if word[-2:] == "es" or word[-2:] == "ed" :
        doubleAndtripple_1 = len(re.findall(r'[eaoui][eaoui]',word))
        if doubleAndtripple_1 > 1 or len(re.findall(r'[eaoui][^eaoui]',word)) > 1 :
            if word[-3:] == "ted" or word[-3:] == "tes" or word[-3:] == "ses" or word[-3:] == "ied" or word[-3:] == "ies" :
                pass
            else :
                disc += 1
 
    #3) discard trailing "e", except where ending is "le"   
    le_except = ['whole','mobile','pole','male','female','hale','pale','tale','sale','aisle','whale','while'] 
    if word[-1:] == "e" :
        if word[-2:] == "le" and word not in le_except :
            pass 
        else :
            disc += 1
 
    #4) check if consecutive vowels exists, triplets or pairs, count them as one. 
    doubleAndtripple = len(re.findall(r'[eaoui][eaoui]',word))
    tripple = len(re.findall(r'[eaoui][eaoui][eaoui]',word))
    disc += doubleAndtripple + tripple
 
    # 5) count remaining vowels in word.
    numVowels = len(re.findall(r'[eaoui]',word))
 
    # 6) add one if starts with "mc"
    if word[:2] == "mc" :
        syls+=1
 
    # 7) add one if ends with "y" but is not surrouned by vowel
    if word[-1:] == "y" and word[-2] not in "aeoui" :
        syls +=1
 
    # 8) add one if "y" is surrounded by non-vowels and is not in the last word.
    for i,j in enumerate(word) :
        if j == "y" :
            if (i != 0) and (i != len(word)-1) :
                if word[i-1] not in "aeoui" and word[i+1] not in "aeoui" :
                    syls+=1
 
    # 9) if starts with "tri-" or "bi-" and is followed by a vowel, add one. 
    if word[:3] == "tri" and word[3] in "aeoui" :
        syls+=1 
    if word[:2] == "bi" and word[2] in "aeoui" :
        syls+=1
 
    # 10) if ends with "-ian", should be counted as two syllables, except for "-tian" and "-cian"
    if word[-3:] == "ian" : 
    # and (word[-4:] != "cian" or word[-4:] != "tian") :
        if word[-4:] == "cian" or word[-4:] == "tian" :
            pass
        else :
            syls+=1
 
    # 11) if starts with "co-" and is followed by a vowel, check if exists in the double syllable dictionary, 
    #     if not, check if in single dictionary and act accordingly. 
    if word[:2] == "co" and word[2] in 'eaoui' :
        if word[:4] in co_two or word[:5] in co_two or word[:6] in co_two :
            syls+=1
        elif word[:4] in co_one or word[:5] in co_one or word[:6] in co_one :
            pass
        else :
            syls+=1
 
    # 12) if starts with "pre-" and is followed by a vowel, check if exists in the double syllable dictionary, 
    #     if not, check if in single dictionary and act accordingly. 
    if word[:3] == "pre" and word[3] in 'eaoui' :
        if word[:6] in pre_one :
            pass
        else :
            syls+=1
 
    # 13) check for "-n't" and cross match with dictionary to add syllable.
    negative = ["doesn't", "isn't", "shouldn't", "couldn't","wouldn't"] 
    if word[-3:] == "n't" :
        if word in negative :
            syls+=1
        else :
            pass  
 
    # 14) Handling the exceptional words. 
    if word in exception_del :
        disc+=1 
    if word in exception_add :
        syls+=1    
 
    # calculate the output
    return numVowels - disc + syls

def get_senses(word):
    """Returns a list of word senses (WordNet synsets) for a word"""
    word_senses = wn.synsets(word)
    return word_senses

def get_synonyms(word_sense):
    synonyms = []
    for lemma in word_sense.lemmas():
        synonym = lemma.name().replace('_', ' ')
        synonyms.append(synonym)
    return synonyms

def number_of_synonyms(word):
    word_senses = get_senses(word)
    num_synonyms = 0
    for i, word_sense in enumerate(word_senses):
        synonyms = get_synonyms(word_sense)
        num_synonyms += len(synonyms) # add number of synonyms of current word sense 
    return num_synonyms

if __name__ == "__main__":
    training_file = "data/complex_words_training.txt"
    development_file = "data/complex_words_development.txt"
    test_file = "data/complex_words_test_unlabeled.txt"

    words_train, labels_train = load_file(training_file)
    words_dev, labels_dev = load_file(development_file)
    
    words_train = words_train + words_dev # concatenate training and dev. sets
    labels_train = labels_train + labels_dev
    
    ngram_counts_file = "ngram_counts.txt.gz"
    counts = load_ngram_counts(ngram_counts_file)
    
    n_words = len(words_train)
    n_features = 4
    X_train = np.zeros((n_words, n_features))
    Y_train = np.asarray(labels_train)
    
    # for i, row in enumerate(X_train):
    #     row[0] = len(words_train[i]) # length of word
    #     row[1] = counts[words_train[i]] # frequency of word in giant corpus
    #     row[2] = count_syllables(words_train[i]) # number of syllables in word
    #     row[3] = number_of_synonyms(words_train[i]) # number of synonyms of word
        
    # mean0, mean1, mean2, mean3 = np.mean(X_train, axis = 0)
    # std0, std1, std2, std3 = np.std(X_train, axis = 0)

    # def normalize_func(X):
    #     feat_0 = ((X[0] - mean0) / std0)
    #     feat_1 = ((X[1] - mean1) / std1)
    #     feat_2 = ((X[2] - mean2) / std2)
    #     feat_3 = ((X[3] - mean3) / std3)
    #     return [feat_0, feat_1, feat_2, feat_3]
    
    # X_train = np.apply_along_axis(normalize_func, 1, X_train)
    
    clf = SVC(gamma='auto')
    clf.fit(X_train, Y_train)

#     Y_pred_train = clf.predict(X_train)
#     tprecision = get_precision(Y_pred_train.tolist(), Y_train.tolist())
#     trecall = get_recall(Y_pred_train.tolist(), Y_train.tolist())
#     tfscore = get_fscore(Y_pred_train.tolist(), Y_train.tolist())
#     training_performance = (tprecision, trecall, tfscore)


    words_test = load_file(development_file)
    
    words_test = []  
    with open(test_file, 'rt', encoding = "utf8") as f:
        i = 0
        for line in f:
            if i > 0: # skips over column headers
                line_split = line[:-1].split("\t")
                words_test.append(line_split[0].lower())
            i += 1
    X_test = np.zeros((len(words_test), n_features))
    for i, row in enumerate(X_test):
        row[0] = len(words_test[i]) # length of word
        row[1] = counts[words_test[i]] # frequency of word in giant corpus
        row[2] = count_syllables(words_test[i]) # number of syllables in word
        row[3] = number_of_synonyms(words_test[i]) # number of synonyms of word

    # X_test = np.apply_along_axis(normalize_func, 1, X_test)
    Y_test = clf.predict(X_test)

    with open('test_labels.txt', 'w') as f:
        for label in Y_test:
            f.write('%s\n' % label)
        
#     dprecision = get_precision(Y_pred_dev.tolist(), Y_dev.tolist())
#     drecall = get_recall(Y_pred_dev.tolist(), Y_dev.tolist())
#     dfscore = get_fscore(Y_pred_dev.tolist(), Y_dev.tolist())
#     development_performance = (dprecision, drecall, dfscore)
