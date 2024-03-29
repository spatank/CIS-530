import re
import nltk
from nltk.tag.perceptron import PerceptronTagger
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

class HearstPatterns(object):
  
  def __init__(self, extended=False):
    self.__chunk_patterns = r""" #  helps us find noun phrase chunks
            NP: {<DT>?<JJ.*>*<NN.*>+}
                {<NN.*>+}
            """
            
    # create a chunk parser
    self.__np_chunker = nltk.RegexpParser(self.__chunk_patterns)

    # now define the Hearst patterns
    # format is <hearst-pattern>, <hypernym_location>
    # so, what this means is that if you apply the first pattern,

    # self.__hearst_patterns = [
    #           ("(NP_[\w\-]+ (, )?especially (NP_[\w\-]+ ? (, )?(and |or )?)+)", "first")
    #       ]
    
    # self.__hearst_patterns = [
    #         ("(NP_[\w\-]+ (, )?such as (NP_[\w\-]+ ? (, )?(and |or )?)+)", "first"),
    #         ("(NP_such_[\w\-]+ as (NP_[\w\-]+ ? (, )?(and |or )?)+)", "first"),
    #         # `such' (JJ) is being merged with the hypernym, workaround above
    #         ("((NP_[\w\-]+ ?(, )?)+(and |or )?NP_other_[\w\-]+)", "second"),
    #         ("(NP_[\w\-]+ (, )?including (NP_[\w\-]+ ? (, )?(and |or )?)+)", "first"),
    #         ("(NP_[\w\-]+ (, )?especially (NP_[\w\-]+ ? (, )?(and |or )?)+)", "first"),
    #     ]

    self.__hearst_patterns = [
            ("(NP_[\w\-]+ (, )?such as (NP_[\w\-]+ ? (, )?(and |or )?)+)", "first"),
            ("(NP_such_[\w\-]+ as (NP_[\w\-]+ ? (, )?(and |or )?)+)", "first"),
            # `such' (JJ) is being merged with the hypernym, workaround above
            ("((NP_[\w\-]+ ?(, )?)+(and |or )?NP_other_[\w\-]+)", "second"),
            ("(NP_[\w\-]+ (, )?including (NP_[\w\-]+ ? (, )?(and |or )?)+)", "first"),
            ("(NP_[\w\-]+ (, )?especially (NP_[\w\-]+ ? (, )?(and |or )?)+)", "first"),
            ("(NP_[\w]+ (, )?has been (NP_[\w]+ ? (, )?(and |or )?)+)", "second"),
            ("(NP_[\w]+ (, )?are otherwise known as (NP_[\w]+ ? (, )?(and |or )?)+)", "first"),
            ("(NP_[\w]+ (, )?is (NP_a_[\w]+ ? (, )?(and |or )?)+)", "second"),
            ("(NP_[\w\-]+ of (NP_[\w\-]+) (, )?includes (NP_[\w\-]+ ? (, )?(and |or )?)+)", "third"),
            ("(there is NP_[\w\-]+ (, )?possibly (NP_[\w\-]+ ? (, )?(and |or )?)+)", "first")
        ]


    if extended:
      self.__hearst_patterns.extend([
            ("(NP_\w+ (, )?such as (NP_\w+ ? (, )?(and |or )?)+)", "first"),
            ])

    self.__pos_tagger = PerceptronTagger()

  def getPatterns(self):
    return self.__hearst_patterns
    
  def prepare(self, rawtext):
    # To process text in NLTK format
    sentences = nltk.sent_tokenize(rawtext.strip())
    sentences = [nltk.word_tokenize(sent) for sent in sentences]
    sentences = [self.__pos_tagger.tag(sent) for sent in sentences]

    return sentences

  def chunk(self, rawtext):
    sentences = self.prepare(rawtext.strip())

    all_chunks = []
    for sentence in sentences:
      chunks = self.__np_chunker.parse(sentence)
      all_chunks.append(self.prepare_chunks(chunks))

    # two or more NPs next to each other should be merged into a single NP,
    # find any N consecutive NP_ and merge them into one...
    # Eg: "NP_foo NP_bar blah blah" becomes "NP_foo_bar blah blah"
    all_sentences = []
    for raw_sentence in all_chunks:
      sentence = re.sub(r"(NP_\w+ NP_\w+)+",
                        lambda m: m.expand(r'\1').replace(" NP_", "_"),
                        raw_sentence)
      all_sentences.append(sentence)

    return all_sentences

  def prepare_chunks(self, chunks):
    # If chunk is NP, start with NP_ and join tokens in chunk with _
    # Else just keep the token as it is

    terms = []
    for chunk in chunks:
      label = None
      try:
        # gross hack to see if the chunk is simply a word or a NP, as
        # we want. But non-NP fail on this method call
        label = chunk.label()
      except:
        pass

      if label is None:  # means one word...
        token = chunk[0]
        terms.append(token)
      else:
        np = "NP_"+"_".join([a[0] for a in chunk])
        terms.append(np)
    
    return ' '.join(terms)

  def find_hyponyms(self, rawtext):
    
    hypo_hypernyms = []
    np_tagged_sentences = self.chunk(rawtext)
    
    for sentence in np_tagged_sentences:
      for (hearst_pattern, parser) in self.__hearst_patterns:
        matches = re.search(hearst_pattern, sentence)
        if matches:
          match_str = matches.group(0)
          nps = [a for a in match_str.split() if a.startswith("NP_")]
          if parser == "first":
            hypernym = nps[0]
            hyponyms = nps[1:]
          elif parser == "third":
            hypernym = nps[0]
            hyponyms = nps[2:]
          else: # usually "second" 
            hypernym = nps[-1]
            hyponyms = nps[:-1]
            
          for i in range(len(hyponyms)):
            hypo_hypernyms.append(
                (self.clean_hyponym_term(hyponyms[i]),
                 self.clean_hyponym_term(hypernym)))

    return hypo_hypernyms

  def clean_hyponym_term(self, term):
    term = term.replace("NP_", "").replace("_", " ").split()[-1]
    return term


if __name__=='__main__':
  hp = HearstPatterns(extended = False)

  text = 'All musical genres, including blues and jazz.'
  hps = hp.find_hyponyms(text)
  print(hps)
  
  text = 'I am going to get green vegetables such as spinach, peas and kale.'
  hps = hp.find_hyponyms(text)
  print(hps)

  text = 'I like to listen to music from musical genres such as blues and jazz.'
  hps = hp.find_hyponyms(text)
  print(hps)

  text = 'works by such authors as Herrick, Goldsmith, and Shakespeare.'
  hps = hp.find_hyponyms(text)
  print(hps)

  text = 'Bruises, wounds, broken bones or other injuries.'
  hps = hp.find_hyponyms(text)
  print(hps)

  text = 'temples, treasuries, and other important civic buildings'
  hps = hp.find_hyponyms(text)
  print(hps)

  text = 'All common-law countries, including Canada and England.'
  hps = hp.find_hyponyms(text)
  print(hps)

  text = 'All musical genres, including blues and jazz.'
  hps = hp.find_hyponyms(text)
  print(hps)

  text = 'most European countries, including Canada and England ...'
  hps = hp.find_hyponyms(text)
  print(hps)
