import math, random
from collections import defaultdict

################################################################################
# Part 0: Utility Functions
################################################################################

COUNTRY_CODES = ['af', 'cn', 'de', 'fi', 'fr', 'in', 'ir', 'pk', 'za']

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

def create_ngram_model(model_class, path, n=2, k=0):
    ''' Creates and returns a new n-gram model trained on the city names
        found in the path file '''
    model = model_class(n, k)
    with open(path, encoding='utf-8', errors='ignore') as f:
        model.update(f.read())
    return model

def create_ngram_model_lines(model_class, path, n=2, k=0):
    ''' Creates and returns a new n-gram model trained on the city names
        found in the path file '''
    model = model_class(n, k)
    with open(path, encoding='utf-8', errors='ignore') as f:
        for line in f:
            model.update(line.strip())
    return model

################################################################################
# Part 1: Basic N-Gram Model
################################################################################

class NgramModel(object):
    ''' A basic n-gram model using add-k smoothing '''

    def __init__(self, n, k):
        self.n = n # order of n-gram model
        self.vocab = set() # initialize vocabulary
        self.context_counts = defaultdict(lambda:0) # frequency of contexts
        self.sequence_counts = defaultdict(lambda:0) # frequency of (context, char) sequences
        self.k = k # smoothing parameter

    def get_vocab(self):
        ''' Returns the set of characters in the vocab '''
        return self.vocab

    def update(self, text):
        ''' Updates the model n-grams based on text '''
        all_ngrams = ngrams(self.n, text)
        for (context, char) in all_ngrams:
            self.vocab.add(char)
            self.context_counts[context] += 1 # increment the context count
            self.sequence_counts[(context, char)] += 1 # increment the (context, character) sequence count

    def prob(self, context, char):
        ''' Returns the probability of char appearing after context '''
        if context in self.context_counts.keys():
            denominator = self.context_counts[context] # frequency of context followed by any token
            numerator = self.sequence_counts[(context, char)] # frequency of exact (context, character) sequence
            prob = (numerator + self.k)/(denominator + (self.k * len(self.vocab)))
            return prob
        else:
            return 1/len(self.vocab)
                
    def random_char(self, context):
        ''' Returns a random character based on the given context and the 
            n-grams learned by this model '''
        r = random.random()
        pre_sum = 0
        for i, char in enumerate(sorted(self.vocab)):
            # pre_sum is sum of probabilities up to, but excluding, the current token
            post_sum = pre_sum + self.prob(context, char)
            # post_sum also includes the probability of the current token
            if pre_sum <= r < post_sum:
                return char
            pre_sum = post_sum
                
    def random_text(self, length):
        output_text = ''
        all_context = start_pad(self.n) # keep a running context list initialized with '~'s
        for i in range(length):
            curr_context = all_context[len(all_context)-self.n:] # extract context from running context list
            next_char = self.random_char(curr_context)
            output_text += next_char
            all_context += next_char
        return output_text

    def perplexity(self, text):
        ''' Returns the perplexity of text based on the n-grams learned by
            this model '''
        m = len(text)
        all_ngrams = ngrams(self.n, text)
        log_prob_sum = 0
        for (context, char) in all_ngrams:
            prob = self.prob(context, char)
            if prob == 0:
                return float('inf')
            log_prob_sum += math.log(prob)
        perplexity = math.exp(-1/(m) * log_prob_sum)
        return perplexity

################################################################################
# Part 2: N-Gram Model with Interpolation
################################################################################

class NgramModelWithInterpolation(NgramModel):
    ''' An n-gram model with interpolation '''

    def __init__(self, n, k):
        self.n = n # highest order n-gram model
        self.models = {} # initialize empty dictionary for NgramModels
        self.weights = {} # lambdas corresponding to each NgramModel
        for order in range(n + 1): # extra model accounts for zeroth order
            self.models[order] = NgramModel(order, k)
            self.weights[order] = 1/(n + 1)
        self.k = k # smoothing parameter
            
    def get_vocab(self):
        vocab = set()
        for order in range(self.n + 1):
            model = self.models[order]
            vocab = vocab.union(model.get_vocab()) # merge vocabularies 
        return vocab

    def update(self, text):
        for order in range(self.n + 1):
            model = self.models[order]
            model.update(text)

    def prob(self, context, char):
        prob = 0
        for order in range(self.n + 1):
            model = self.models[order]
            weight = self.weights[order]
            if model.n == 0:
                sliced_context = ''
            else:
                sliced_context = context[-model.n:]
            prob += weight * model.prob(sliced_context, char)
        return prob

################################################################################
# Part 3: Your N-Gram Model Experimentation
################################################################################

if __name__ == '__main__':
    pass