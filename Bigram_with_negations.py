# NaiveBayes

import csv
import sys
import getopt
import os
import math
import random

"""
=========================Naive Bayes Classifier===========================================================================
Probability P(c|d)= P(d|c)*P(c)/P(d)
based on which probability P(c1|d) or P(c2|d), class c1(pos) or c2(neg) is chosen...
P(c) is the prior probability ( P(c)=0.5 ) for our training corpus (containing equal no. of +ve and -ve reviews)
P(d|c) is calculated as:
    P(d|c)=P(w1,w2,....w(n)|c)
          =P(w1,w2|c)*P(w2,w3|c)....*P(w(n-1),w(n)|c)   (Assumption is all conditional probabilities are independent...)
          For Bigram model...
          P(w1,w2|c)=(count(w1,w2|c)+1)/(count(c)+|V|)   (used laplace add 1 smoothing)
          For Unigram model...
            Probablity of P(w|c)=(count(w,c)+1)/(count(c)+|V|)
                count(w,c) = count of word w in all documents of class c
                count(c)   = count of words in class c
                |V|        = total distint words in the our trainig set
               used add one smoothing
===========================================================================================================================         
"""

class TrainSplit:
    """Represents a set of training/testing data. self.train is a list of Examples, as is self.test. 
    """
    def __init__(self):
          self.train = []
          self.test = []

class Example:
    """Represents a document with a label. klass is 'pos' or 'neg' by convention.
       words is a list of strings.
    """
    def __init__(self):
          self.klass = ''
          self.words = []
class NaiveBayes():
    """ Class to implement the Naive Bayes model...
    """
    trainDir="data/imdb1"
    singleTestDir="data"
    megaPos={}
    megaNeg={}
    vocabularySizePos=0
    vocabularySizeNeg=0
    totalVocabularySize=0
    punc=",. !:\t\n"
    numfolds=10
    #function to read the training/test data from files...
    
    def readFile(self, fileName):
        contents = []
        f = open(fileName)
        for line in f:
            contents.append(line)
        f.close()
        result = self.segmentWords('\n'.join(contents))    
        return result
    
    #function to segment the words read from file...
    def segmentWords(self,t):
        s=t
        for ch in self.punc:
            s=s.replace(ch," ")
        s=self.applyNegators(s.split(),t.split())
        return s
    
    #function to apply negtors in the words scanned/parsed...
    def applyNegators(self,s,t):
        #get the negators list...
        f=open("negators.txt","r");
        _list=f.read()
        _list=_list.replace("\n"," ")
        _list=_list.split()
        #apply the negators...
        size=len(s);i=0;j=0;
        while i<size:
            if s[i] in _list:
                del s[i];j+=1;
                size-=1
                while True:
                    if(i==size):
                        break
                    if s[i] in _list:
                        del s[i];
                        size-=1
                    else:
                        s[i]+="-NOT";i+=1;
                    b=False
                    for ch in self.punc:
                        if ch in t[j]:
                            b=True
                            break
                    j+=1;
                    if b==True:
                        break
            else:
                i+=1;
        return s
    
    #main driver-function to perform cross-validation on the whole training set in 10 folds...
    def buildSplits(self):
        posTrainFileNames = os.listdir('%s/pos/' % self.trainDir)
        negTrainFileNames = os.listdir('%s/neg/' % self.trainDir)
        AvgAcc=0.0
        for fold in range(0,self.numfolds):
            print "[INFO]\tPerforming cross-validation on fold: "+`fold`
            split = TrainSplit()
            for fileName in posTrainFileNames:
              example = Example()
              example.words = self.readFile('%s/pos/%s' % (self.trainDir, fileName))
              example.klass = 'pos'
              if fileName[2] == str(fold):
                split.test.append(example)
              else:
                split.train.append(example)
            for fileName in negTrainFileNames:
              example = Example()
              example.words = self.readFile('%s/neg/%s' % (self.trainDir, fileName))
              example.klass = 'neg'
              if fileName[2] == str(fold):
                split.test.append(example)
              else:
                split.train.append(example)
            ac=self.TrainAndTest(split)
            print 'accuracy: %.2f %s' % (ac,'%')
            AvgAcc+=ac
        print "overallaccuracy:\t %.2f %s" % (AvgAcc/10.0,"%")
            
    #Training and Testing the data...
    def TrainAndTest(self,split):
        self.Train(split)
        return self.Test(split)
    def Train(self,split):
        self.megaPos={}
        self.megaNeg={}
        for ex in split.train:
            words=ex.words
            if ex.klass=='pos':
                for i in range(0,len(words)-1):
                    pair=(words[i],words[i+1])
                    if pair in self.megaPos:
                        self.megaPos[pair]+=1
                    else:
                        self.megaPos[pair]=1
            else:
                for i in range(0,len(words)-1):
                    pair=(words[i],words[i+1])
                    if pair in self.megaNeg:
                        self.megaNeg[pair]+=1
                    else:
                        self.megaNeg[pair]=1
        for key in self.megaPos:
            self.vocabularySizePos+=self.megaPos[key]
        for key in self.megaNeg:
            self.vocabularySizeNeg+=self.megaNeg[key]
        self.totalVocabularySize=self.vocabularySizePos+self.vocabularySizeNeg
    def Test(self,split):
        accuracy=0.0
        for ex in split.test:
            words=ex.words
            if self.classify(words,"db")==ex.klass:
                accuracy+=1.0
        accuracy/=len(split.test)
        accuracy*=100.0
        return accuracy


    def TrainOnWholeCorpus(self):
        posTrainFileNames = os.listdir('%s/pos/' % self.trainDir)
        negTrainFileNames = os.listdir('%s/neg/' % self.trainDir)
        split = TrainSplit()
        for fileName in posTrainFileNames:
            example = Example()
            example.words = self.readFile('%s/pos/%s' % (self.trainDir, fileName))
            example.klass = 'pos'
            split.train.append(example)
        for fileName in negTrainFileNames:
              example = Example()
              example.words = self.readFile('%s/neg/%s' % (self.trainDir, fileName))
              example.klass = 'neg'
              split.train.append(example)
        self.Train(split)

        """print values in a file
        """
        w = csv.writer(open("megaPos.csv", "w"))
        for key, val in self.megaPos.items():
            w.writerow([key, val])
        w = csv.writer(open("megaNeg.csv", "w"))
        for key, val in self.megaNeg.items():
            w.writerow([key, val])
        print self.vocabularySizePos
        print self.vocabularySizeNeg
        print self.totalVocabularySize

    
    #classifier based on Naive Bayes...
    def classify(self,words,_type):
        print words;
        posProb=math.log(0.5)
        for i in range(0,len(words)-1):
            pair=(words[i],words[i+1])
            if pair in self.megaPos:
                posProb+=math.log((self.megaPos[pair]+1.0)/(self.totalVocabularySize + self.vocabularySizePos));
            else:
                posProb+=math.log((1.0)/(self.totalVocabularySize + self.vocabularySizePos));
        negProb=math.log(0.5)
        for i in range(0,len(words)-1):
            pair=(words[i],words[i+1])
            if pair in self.megaNeg:
                negProb+=math.log((self.megaNeg[pair]+1.0)/(self.totalVocabularySize + self.vocabularySizeNeg));
            else:
                negProb+=math.log((1.0)/(self.totalVocabularySize + self.vocabularySizeNeg));
        if(posProb>=negProb):
            if _type=="sen":
                print "rating:\t%.0f" % ((-posProb/len(words))/5.0+2.0)
            else:
                #print "rating:\t%.0f" % ((-posProb/len(words))-12+2.0)
                pass
            return "pos"
        else:
            if _type=="sen":
                print "rating:\t%.0f" % ((-posProb/len(words))/5.0-1)
            else:
                #print "rating:\t%.0f" % ((-posProb/len(words))-12)
                pass
            return "neg"

    #init function to run the main-driver function
    def __init__(self):
        #self.buildSplits()
        pass

def classify(_str):
    nb=NaiveBayes()
    if len(_str.split())==1:
        _str="was "+_str
    if nb.classify(nb.segmentWords(_str),"sen")=="pos":
        return 'pos' 
    else:
        return 'neg' 
def main():
    nb=NaiveBayes()
    nb.TrainOnWholeCorpus();
if __name__ == "__main__":
    main()
