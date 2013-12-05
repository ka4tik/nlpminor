import sys
#import Bigram_with_negations

def lines(fp):
    print str(len(fp.readlines()))

def getsenti(inputtext):
    scores = {} # initialize an empty dictionary
    vfile = open("values.txt")
    for line in vfile:
      term, score  = line.split("\t")  
      scores[term] = int(score)  # convert the score to an integer.
    words=inputtext.split();
    senti=0;
    for w in words:
        if w in scores.keys():
            senti=senti+scores[w];
    return senti
