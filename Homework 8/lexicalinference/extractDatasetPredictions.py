import os
import pprint
import argparse

pp = pprint.PrettyPrinter()
parser = argparse.ArgumentParser()

parser.add_argument('--extractionsfile', type=str, required=True)
parser.add_argument('--trdata', type=str, required=True)
parser.add_argument('--valdata', type=str, required=True)
parser.add_argument('--testdata', type=str, required=True)

parser.add_argument('--trpredfile', type=str, required=True)
parser.add_argument('--valpredfile', type=str, required=True)
parser.add_argument('--testpredfile', type=str, required=True)


def convertExtractionsToDict(extractionsfile):

    hyper2hypos = {}

    with open(extractionsfile, 'r') as f:
        text = f.read().strip().split('\n')
        for line in text:
            hypo, hyper = line.split('\t')
            hypo, hyper = hypo.lower(), hyper.lower()
            if hyper not in hyper2hypos:
                hyper2hypos[hyper] = set()
            hyper2hypos[hyper].add(hypo)

    return hyper2hypos


def writeDataPredictions(hyper2hypos, datafile, predfile):
    directory = os.path.dirname(predfile)
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(datafile, 'r') as f:
        inputlines = f.read().strip().split('\n')

    predictions = []

    for line in inputlines:
        linesplit = line.split('\t')
        hypo, hyper = linesplit[0], linesplit[1]
        if hyper not in hyper2hypos:
            predictions.append('{}\t{}\tFalse'.format(hypo, hyper))
        else:
            if hypo not in hyper2hypos[hyper]:
                predictions.append('{}\t{}\tFalse'.format(hypo, hyper))
            else:
                predictions.append('{}\t{}\tTrue'.format(hypo, hyper))

    with open(predfile, 'w') as f:
        for line in predictions:
            f.write(line)
            f.write('\n')


def main(args):
    hyper2hypos = convertExtractionsToDict(args.extractionsfile)
    writeDataPredictions(hyper2hypos, args.trdata, args.trpredfile)
    writeDataPredictions(hyper2hypos, args.valdata, args.valpredfile)
    writeDataPredictions(hyper2hypos, args.testdata, args.testpredfile)


if __name__ == '__main__':
    args = parser.parse_args()
    pp.pprint(args)
    main(args)
