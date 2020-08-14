import pprint
import argparse

pp = pprint.PrettyPrinter()
parser = argparse.ArgumentParser()

parser.add_argument('--goldfile', type=str, required=True)
parser.add_argument('--predfile', type=str, required=True)


def readLabels(datafile):
    ''' Datafile contains 3 columns,
        col1: hyponym
        col2: hypernym
        col3: label
    '''

    labels = []
    with open(datafile, 'r') as f:
        inputlines = f.read().strip().split('\n')

    for line in inputlines:
        hypo, hyper, label = line.split('\t')
        if label == 'True':
            labels.append(1)
        else:
            labels.append(0)

    return labels


def computePRF(truthlabels, predlabels):
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    for t, p in zip(truthlabels, predlabels):
        if t == 1:
            if p == 1:
                tp += 1
            else:
                fn += 1
        else:
            if p == 1:
                fp += 1
            else:
                tn += 1

    prec = float(tp)/float(tp + fp)
    recall = float(tp)/float(tp + fn)
    f1 = 2*prec*recall/(prec + recall)

    print("Precision:{} Recall:{} F1:{}".format(prec, recall, f1))

    return prec, recall, f1


def main(args):
    gold = readLabels(args.goldfile)
    pred = readLabels(args.predfile)

    print("Performance")
    computePRF(gold, pred)


if __name__ == '__main__':
    args = parser.parse_args()
    pp.pprint(args)
    main(args)
