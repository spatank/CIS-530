#!/usr/bin/env python

'''
predict_ext.py

Use this file to implement your extended model.

Usage:
python predict_lm.py <SPLIT> <CONDITION_POS_MODEL> <CONDITION_NEG_MODEL> <OUTFILE>

You may add or remove arguments as necessary; include a description in the README.
'''

from collections import *
import json
import os, sys
import operator

import util

def uniform_lambdas(n):
    lambdas = {}
    for order in range(n + 1):
        lambdas[order] = 1/(n + 1)
    return lambdas

import numpy as np

def increasing_lambdas(n):
    lambdas = {}
    lambdas_array = np.linspace(0, 1, n + 1)
    lambdas_array = lambdas_array/sum(lambdas_array) # ensure sum is 1
    for order, weight in enumerate(lambdas_array):
        lambdas[order] = weight
    return lambdas

def decreasing_lambdas(n):
    lambdas = {}
    lambdas_array = np.linspace(0, 1, n + 1)
    lambdas_array = np.flip(lambdas_array/sum(lambdas_array)) # ensure sum is 1
    for order, weight in enumerate(lambdas_array):
        lambdas[order] = weight
    return lambdas

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


def score_subjects(data, pos_models, neg_models):

    subj2tweets = {}
    for subjID, subjData, subjTweets in data:
        tweets = util.get_tweets_element(subjTweets, elem='text')
        subj2tweets[subjID] = tweets
    
    scores = {}
    ##### BEGIN SOLUTION #####
    weights = uniform_lambdas(len(pos_models) - 1)

    for subjID, tweets in subj2tweets.items():
        user_scores = []
        for idx, tweet in enumerate(tweets):
            if idx % 10 == 0:
                C = 0
                num = 0
                prob_pos = 0
                prob_neg = 0
                for idx in range(len(pos_models)):
                    lm_pos = pos_models[idx]
                    lm_neg = neg_models[idx]
                    weight = weights[idx]
                    grams = ngrams(idx, tweet)
                    for (history, char) in grams:
                        if history in lm_pos:
                            if char in lm_pos[history]:
                                if history in lm_neg:
                                    if char in lm_neg[history]:
                                        prob_pos += np.log(weight * lm_pos[history][char])
                                        prob_neg += np.log(weight * lm_neg[history][char])
                                        num = prob_pos - prob_neg
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
    condPOSmodels_paths = sys.argv[2]
    condPOSmodels = os.listdir(condPOSmodels_paths)
    condNEGmodels_paths = sys.argv[3]
    condNEGmodels = os.listdir(condNEGmodels_paths)
    outfile = sys.argv[4]
    
    data = util.load_data(split)

    pos_models = {}
    neg_models = {}

    for pos_model in condPOSmodels:
        if pos_model.startswith('.'):
            continue
        model_path = os.path.join(condPOSmodels_paths + '/' + pos_model)
        print(model_path)
        model, order = util.load_lm(model_path)
        pos_models[order] = model

    for neg_model in condNEGmodels:
        if neg_model.startswith('.'):
            continue
        model_path = os.path.join(condNEGmodels_paths + '/' + neg_model)
        print(model_path)
        model, order = util.load_lm(model_path)
        neg_models[order] = model

    
    subject_scores = score_subjects(data, pos_models, neg_models)
    
    write_rankings(subject_scores, outfile)
    
