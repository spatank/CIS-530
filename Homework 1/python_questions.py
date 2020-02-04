'''Homework 1 Python Questions

This is an individual homework
Implement the following functions.

Do not add any more import lines to this file than the ones
already here without asking for permission on Piazza.
Use the regular expression tools built into Python; do NOT use bash.
'''

import re

def check_for_foo_or_bar(text):
  p_1 = re.compile(r'\b[fF]oo\b')
  p_2 = re.compile(r'\b[bB]ar\b')
  if p_1.search(text) and p_2.search(text):
      return True
  return False

def replace_rgb(text):
  # look for hexadecimal colors between 3 and 6 long 
  p_1 = r'(?<!\S)#[\d][a-zA-Z0-9\d]{2,5}(?!\S)'
  # look for rgb colors, decimal or otherwise
  p_2 = r'(?<!\S)rgb\([\d]{1,3}[\.]?[\d]?,[\s]?[\d]{1,3}[\.]?[\d]?,[\s]?[\d]{1,3}[\.]?[\d]?\)(?!\S)'
  replacements = [(p_1, 'COLOR'), (p_2, 'COLOR')]
  for pattern, replace_with in replacements:
    text = re.sub(pattern, replace_with, text)
  return text

def edit_distance(str_1, str_2):
    rows = len(str_1) + 1
    cols = len(str_2) + 1
    dist = []
    for i in range(rows):
        row = []
        for j in range(cols):
            row.append(0)
        dist.append(row)
    for i in range(1, rows):
        dist[i][0] = i
    for j in range(1, cols):
        dist[0][j] = j
    for j in range(1, cols):
        for i in range(1, rows):
            if str_1[i - 1] == str_2[j - 1]:
                cost = 0
            else:
                cost = 1
            dist[i][j] = min(dist[i-1][j] + 1, 
                                 dist[i][j-1] + 1, 
                                 dist[i-1][j-1] + cost) 
    return dist[len(str_1)][len(str_2)]


def wine_text_processing(wine_file_path, stopwords_file_path):
  '''Process the two files to answer the following questions and output results to stdout.

  1. What is the distribution over star ratings?
  2. What are the 10 most common words used across all of the reviews, and how many times
     is each used?
  3. How many times does the word 'a' appear?
  4. How many times does the word 'fruit' appear?
  5. How many times does the word 'mineral' appear?
  6. Common words (like 'a') are not as interesting as uncommon words (like 'mineral').
     In natural language processing, we call these common words "stop words" and often
     remove them before we process text. stopwords.txt gives you a list of some very
     common words. Remove these stopwords from your reviews. Also, try converting all the
     words to lower case (since we probably don't want to count 'fruit' and 'Fruit' as two
     different words). Now what are the 10 most common words across all of the reviews,
     and how many times is each used?
  7. You should continue to use the preprocessed reviews for the following questions
     (lower-cased, no stopwords).  What are the 10 most used words among the 5 star
     reviews, and how many times is each used? 
  8. What are the 10 most used words among the 1 star reviews, and how many times is
     each used? 
  9. Gather two sets of reviews: 1) Those that use the word "red" and 2) those that use the word
     "white". What are the 10 most frequent words in the "red" reviews which do NOT appear in the
     "white" reviews?
  10. What are the 10 most frequent words in the "white" reviews which do NOT appear in the "red"
      reviews?

  No return value.
  '''

  pass

