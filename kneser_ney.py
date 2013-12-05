# NaiveBayes

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
    PcontinuationPos={}
    PcontinuationNeg={}
    lambdaPos={}
    lambdaNeg={}
    uniMegaPos={}
    uniMegaNeg={}
    punc=",. !:\t\n"
    numfolds=10
    discount=0.01
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
        for i in range(0,len(s)):
            if s[i] in _list:
                i+=1
                if i==len(s):
                    break
                while True:
                    s[i]+="-NOT"
                    i+=1
                    if(i==len(s)):
                        break
                    b=False
                    for ch in self.punc:
                        if ch in t[i]:
                            b=True
                            break
                    if b==True:
                        break
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
        self.uniMegaPos={}
        self.uniMegaNeg={}
        for ex in split.train:
            words=ex.words
            if ex.klass=='pos':
                for word in words:
                    if word in self.uniMegaPos:
                        self.uniMegaPos[word]+=1
                    else:
                        self.uniMegaPos[word]=1
                
                for i in range(0,len(words)-1):
                    pair=(words[i],words[i+1])
                    if pair in self.megaPos:
                        self.megaPos[pair]+=1
                    else:
                        self.megaPos[pair]=1
            else:
                for word in words:
                    if word in self.uniMegaNeg:
                        self.uniMegaNeg[word]+=1
                    else:
                        self.uniMegaNeg[word]=1
                
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
        self.PcontinuationPos={}
        self.PcontinuationNeg={}
        for p in self.megaPos:
            if p[1] in self.PcontinuationPos:
                self.PcontinuationPos[p[1]]+=1
            else:
                self.PcontinuationPos[p[1]]=1
        for p in self.megaNeg:
            if p[1] in self.PcontinuationNeg:
                self.PcontinuationNeg[p[1]]+=1
            else:
                self.PcontinuationNeg[p[1]]=1
        for w in self.PcontinuationPos:
            self.PcontinuationPos[w]/=len(self.megaPos)
        for w in self.PcontinuationNeg:
            self.PcontinuationNeg[w]/=len(self.megaNeg)
        self.lambdaPos={}
        self.lambdaNeg={}
        for p in self.megaPos:
            if p[0] in self.lambdaPos:
                self.lambdaPos[p[0]]+=1
            else:
                self.lambdaPos[p[0]]=1
        for p in self.megaNeg:
            if p[0] in self.lambdaNeg:
                self.lambdaNeg[p[0]]+=1
            else:
                self.lambdaNeg[p[0]]=1
        d=self.discount;
        for w in self.lambdaPos:
            self.lambdaPos[w]*=(d/self.uniMegaPos[w]);
        for w in self.lambdaNeg:
            self.lambdaNeg[w]*=(d/self.uniMegaNeg[w]);
        
        
    def Test(self,split):
        accuracy=0.0
        for ex in split.test:
            words=ex.words
            if self.classify(words)==ex.klass:
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

    
    #classifier based on Naive Bayes...
    def classify(self,words):
        posProb=math.log(0.5)
        d=self.discount;
        for i in range(0,len(words)-1):
            pair=(words[i],words[i+1])
            if (pair in self.megaPos):
                posProb+=math.log( max(self.megaPos[pair]-d,0)/self.uniMegaPos[pair[0]] +  (self.lambdaPos[pair[0]]*self.PcontinuationPos[pair[1]]) );
            else:
                pass
                #posProb+=math.log(self.lambdaPos[pair[0]]*self.PcontinuationPos[pair[1]]);
        negProb=math.log(0.5)
        for i in range(0,len(words)-1):
            pair=(words[i],words[i+1])
            if (pair in self.megaNeg):
                posProb+=math.log(max(self.megaNeg[pair]-d,0)/self.uniMegaNeg[pair[0]] +  (self.lambdaNeg[pair[0]]*self.PcontinuationNeg[pair[1]]));
            else:
                pass
                #posProb+=math.log(self.lambdaNeg[pair[0]]*self.PcontinuationNeg[pair[1]]);
        if(posProb>=negProb):
            return "pos"
        else:
            return "neg"

    #init function to run the main-driver function
    def __init__(self):
        self.buildSplits()
        pass

#print "hello"
nb=NaiveBayes()
nb.TrainOnWholeCorpus()
"""print nb.classify(nb.segmentWords("bad"));
print nb.classify(nb.segmentWords("not bad"));
print nb.classify(nb.segmentWords("awesome"));
print nb.classify(nb.segmentWords("good"));
print nb.classify(nb.segmentWords("not good"));
print nb.classify(nb.segmentWords("was"));
print nb.classify(nb.segmentWords("movie was awesome"));
print nb.classify(nb.segmentWords("movie was horrible"));
print nb.classify(nb.segmentWords("movie was awesome but acting was bad"));
print nb.classify(nb.segmentWords("not impressive"));"""

#taking inputs from the user (prompt on console)...
_str=raw_input("Enter sentences\n");
while _str!="":
    if len(_str.split())==1:
        _str="was "+_str
    if nb.classify(nb.segmentWords(_str))=="pos":
        print "positive sentiment"
    else:
        print "negative sentiment"
    _str=raw_input("");
    
print nb.classify(nb.readFile('rt-polaritydata/rt-polarity.pos'));
print nb.classify(nb.readFile('rt-polaritydata/rt-polarity.neg'));
