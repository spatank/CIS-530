'''
predict_lm.py

Predict the condition of subjects based on their tweets and pre-trained
language models.

The script takes as input:
- a SPLIT (can be one of train/dev/test)
- a CONDITION_POS language model file (trained using train_lm.py)
- a CONDITION_NEG language model file
- an OUTFILE (location to write the results)

The script should write to OUTFILE an ordered list of subjects from
the chosen dataset split, where subjects at the top of the list
are most likely to have CONDITION_POS and those at the bottom are
most likely to have CONDITION_NEG based on their tweets and
the two language models.

Usage:

python predict_lm.py <SPLIT> <CONDITION_POS_MODEL> <CONDITION_NEG_MODEL> <OUTFILE>
'''


from collections import *
import tarfile
import numpy as np 
import json
import os, sys
import codecs
import operator
import statistics
import math

import util

def start_pad(n):
    ''' Returns a padding string of length n to append to the front of text
        as a pre-processing step to building n-grams '''
    return '~' * n

def ngrams(n, text):
    ''' Returns the ngrams of the text as tuples where the first element is
        the length-n context and the second is the character '''
    text = start_pad(n) + text
    grams = []
    for j in range(len(text)-n):
        context = text[j:j+n]
        char = text[j+n]
        grams.append((context, char))
    return grams


def score_subjects(data, lm_pos, lm_neg):
    '''
    Score the persons represented in data on the basis of their
    tweet text, and the language models lm_pos and lm_neg.
    
    :param data: list of tuples containing (subjectID, subject metadata, subject tweets)
    :param lm_pos: dict; language model corresponding to positive condition (as output by train_lm.py)
    :param lm_neg: dict; language model corresponding to negative condition (as output by train_lm.py)
    
    :returns dict of {subjectID: score} where scores correspond to the likelihood that
             each subject has the positive condition as opposed to the negative condition
    '''
    
    # create a dict mapping subjectID to tweet text
    subj2tweets = {}
    for subjID, subjData, subjTweets in data:
        tweets = util.get_tweets_element(subjTweets, elem='text')
        subj2tweets[subjID] = tweets
    
    scores = {}
    ##### BEGIN SOLUTION #####
    order_pos = len(list(lm_pos.keys())[0])
    order_neg = len(list(lm_neg.keys())[0])

    for subjID, tweets in subj2tweets.items():
        user_scores = []
        for idx, tweet in enumerate(tweets):
            if idx % 10 == 0:
                C = 0
                grams = ngrams(order_pos, tweet)
                num = 0
                for (history, char) in grams:
                    if history in lm_pos:
                        if history in lm_neg:
                            if char in lm_pos[history]:
                                if char in lm_neg[history]:
                                    prob_pos = lm_pos[history][char]
                                    prob_neg = lm_neg[history][char]
                                    num += (np.log(prob_pos) - np.log(prob_neg))
                                    C += 1
                if C == 0:
                    continue
                user_scores.append(num/C)
        scores[subjID] = np.median(user_scores)
    ##### END SOLUTION #####
    
    return scores

def write_rankings(user_scores, writeFile):
    sorted_dict = sorted(user_scores.items(), key=operator.itemgetter(1), reverse=True)
    with open(writeFile, 'w') as out:
        for user, score in sorted_dict:
            try:
                out.write(user + '\n')
            except:
                pass

if __name__ == '__main__':
    
    split = sys.argv[1]
    condPOSmodel = sys.argv[2]
    condNEGmodel = sys.argv[3]
    outfile = sys.argv[4]
    
    data = util.load_data(split)
    
    lm_pos, order_pos = util.load_lm(condPOSmodel)
    lm_neg, order_neg = util.load_lm(condNEGmodel)
    
    assert order_pos == order_neg
    
    subject_scores = score_subjects(data, lm_pos, lm_neg)
    
    write_rankings(subject_scores, outfile)
    
