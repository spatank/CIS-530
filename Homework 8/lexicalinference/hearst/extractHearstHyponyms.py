import os
import pprint
import argparse
from hearstPatterns import HearstPatterns

pp = pprint.PrettyPrinter()
parser = argparse.ArgumentParser()

parser.add_argument('--inputwikifile', type=str, required=True)
parser.add_argument('--outputfile', type=str, required=True)


def extractHearstPatterns(inputwikifile, hearstPatterns):
    '''Each line in inputwikifile contains 2 columns
        col1: tokenized text
        col2: tokenized lemmatized text
    '''

    # Should contain list of (hyponym, hypernym) tuples
    hearstExtractions = []
    lines_read = 0

    with open(inputwikifile, 'r') as f:
        for line in f:
            lines_read += 1
            line = line.strip()
            if not line:
                continue
            line_split = line.split("\t")
            sentence, lemma_sent = line_split[0].strip(), line_split[1].strip()

            hypo_hyper_pairs = hearstPatterns.find_hyponyms(sentence)
            # hypo_hyper_pairs = hearstPatterns.find_hyponyms(lemma_sent)

            hearstExtractions.extend(hypo_hyper_pairs)

            if lines_read % 10000 == 0:
                print("Lines Read: {}".format(lines_read))

    return hearstExtractions


def writeHypoHyperPairsToFile(hypo_hyper_pairs, outputfile):
    directory = os.path.dirname(outputfile)
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(outputfile, 'w') as f:
        for (hypo, hyper) in hypo_hyper_pairs:
            f.write(hypo + "\t" + hyper + '\n')


def main(args):
    print(args.inputwikifile)
    hearstPatterns = HearstPatterns(extended=False)

    hypo_hyper_pairs = extractHearstPatterns(args.inputwikifile,
                                             hearstPatterns)

    writeHypoHyperPairsToFile(hypo_hyper_pairs, args.outputfile)


if __name__ == '__main__':
    args = parser.parse_args()
    pp.pprint(args)
    main(args)
