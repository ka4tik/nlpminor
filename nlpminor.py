#!/usr/bin/python
# -*- coding: iso-8859-15 -*-
import cgi

from google.appengine.api import users
import webapp2
import csv
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

prob=0.00;
rating=0.00;
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

        """instead of training load values
        """

        self.megaPos = {}
        for key, val in csv.reader(open("megaPos.csv")):
            self.megaPos[key] = val
        self.megaNeg = {}
        for key, val in csv.reader(open("megaNeg.csv")):
            self.megaNeg[key] = val

        self.vocabularySizePos=703675
        self.vocabularySizeNeg=669053
        self.totalVocabularySize=1372728
        pass

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
                rating=((-posProb/len(words))/5.0+2.0)
                prob=-posProb
            else:
                #print "rating:\t%.0f" % ((-posProb/len(words))-12+2.0)
                pass
            return "pos"
        else:
            if _type=="sen":
                print "rating:\t%.0f" % ((-negProb/len(words))/5.0)
                rating=((-negProb/len(words))/5.0)
                prob=-negProb;
            else:
                #print "rating:\t%.0f" % ((-posProb/len(words))-12)
                pass
            return "neg"

    def getrating(self,words,_type):
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
            rating=((-posProb/len(words))/5.0+2.0)
            return math.floor(rating)
        else:
            rating=((-negProb/len(words))/5.0)
            return math.floor(rating);
    #init function to run the main-driver function
    def __init__(self):
        #self.buildSplits()
        pass

nb=NaiveBayes()
def TrainOnPageLoad():
    nb.TrainOnWholeCorpus();
def classify(_str):
    if len(_str.split())==1:
        _str="was "+_str
    if nb.classify(nb.segmentWords(_str),"sen")=="pos":
        return ('pos',nb.getrating(nb.segmentWords(_str),"sen"));
    else:
        return ('neg',nb.getrating(nb.segmentWords(_str),"sen"));

MAIN_PAGE_FANCY="""\
<html>
<head>
    <style type="text/css">
    body {background-color:grey;}
    h1 { font-size:200%;color:black;}
    p { font-size:120%;color:blue;}
    #left{
        position:fixed;
        right:100px;
        top:200px;
    }
    .textbox{
    background-color : #99FFCC;
	border: 2px solid #008000;
    }
    input#gobutton{
    cursor:pointer; /*forces the cursor to change to a hand when the button is hovered*/
    padding:5px 25px; /*add some padding to the inside of the button*/
    background:#35b128; /*the colour of the button*/
    border:1px solid #33842a; /*required or the default border for the browser will appear*/
    /*give the button curved corners, alter the size as required*/
    -moz-border-radius: 10px;
    -webkit-border-radius: 10px;
    border-radius: 10px;
    /*give the button a drop shadow*/
    -webkit-box-shadow: 0 0 4px rgba(0,0,0, .75);
    -moz-box-shadow: 0 0 4px rgba(0,0,0, .75);
    box-shadow: 0 0 4px rgba(0,0,0, .75);
    /*style the text*/
    color:#f3f3f3;
    font-size:1.1em;
    }
    /***NOW STYLE THE BUTTON'S HOVER AND FOCUS STATES***/
    input#gobutton:hover, input#gobutton:focus{
    background-color :#399630; /*make the background a little darker*/
    /*reduce the drop shadow size to give a pushed button effect*/
    -webkit-box-shadow: 0 0 1px rgba(0,0,0, .75);
    -moz-box-shadow: 0 0 1px rgba(0,0,0, .75);
    box-shadow: 0 0 1px rgba(0,0,0, .75);
    }
    

    /***FIRST STYLE THE BUTTON***/
    input#bigbutton {
    width:500px;
    background: #3e9cbf; /*the colour of the button*/
    padding: 8px 14px 10px; /*apply some padding inside the button*/
    border:1px solid #3e9cbf; /*required or the default border for the browser will appear*/
    cursor:pointer; /*forces the cursor to change to a hand when the button is hovered*/
    /*style the text*/
    font-size:1.5em;
    font-family:Oswald, sans-serif; /*Oswald is available from http://www.google.com/webfonts/specimen/Oswald*/
    letter-spacing:.1em;
    text-shadow: 0 -1px 0px rgba(0, 0, 0, 0.3); /*give the text a shadow - doesn't appear in Opera 12.02 or earlier*/
    color: #fff;
    /*use box-shadow to give the button some depth - see cssdemos.tupence.co.uk/box-shadow.htm#demo7 for more info on this technique*/
    -webkit-box-shadow: inset 0px 1px 0px #3e9cbf, 0px 5px 0px 0px #205c73, 0px 10px 5px #999;
    -moz-box-shadow: inset 0px 1px 0px #3e9cbf, 0px 5px 0px 0px #205c73, 0px 10px 5px #999;
    box-shadow: inset 0px 1px 0px #3e9cbf, 0px 5px 0px 0px #205c73, 0px 10px 5px #999;
    /*give the corners a small curve*/
    -moz-border-radius: 10px;
    -webkit-border-radius: 10px;
    border-radius: 10px;
    }
    /***SET THE BUTTON'S HOVER AND FOCUS STATES***/
    input#bigbutton:hover, input#bigbutton:focus {
    color:#dfe7ea;
    /*reduce the size of the shadow to give a pushed effect*/
    -webkit-box-shadow: inset 0px 1px 0px #3e9cbf, 0px 2px 0px 0px #205c73, 0px 2px 5px #999;
    -moz-box-shadow: inset 0px 1px 0px #3e9cbf, 0px 2px 0px 0px #205c73, 0px 2px 5px #999;
    box-shadow: inset 0px 1px 0px #3e9cbf, 0px 2px 0px 0px #205c73, 0px 2px 5px #999;
    }
    #italic
    {
        font-style:italic;
    }
    #oblique
    {
        font-style:oblique;
        color:red;
    }
    img{
        width: 180px;
        height: auto;
    }


    </style>
    </head>
    <body>
    <img id="left" src="/images/logo.png">
    <h1 id="italic">NLP minor project</h1>
    <p> Enter Text Below : </p>
    <form action="/sign" method="post">
      <div><textarea class="textbox" name="content" rows="25" cols="120"></textarea></div>
      <div><input id="gobutton" type="submit" value="Get Sentiment"></div>
      <!--<div><input id="bigbutton" type="submit" value="Get Sentiment"></div>-->
    </form>
    <p id="oblique">By: 
    Aditya Gaurav, 
    Aman Gupta,
    Ashish Yadav,
    Kartik Singal</p>
  </body>
</html>
"""

MAIN_PAGE_HTML = """\
<html>
<head>
    <style type="text/css">
    body {background-color:grey;}
    h1 { font-size:200%;color:black;}
    p { font-size:120%;color:blue;}
    </style>
    </head>
    <body>
    <img src="/images/logo.png">
    <form action="/sign" method="post">
    <h1>NLP minor project</h1>
    <p> Enter Text Below : </p>
      <div><textarea name="content" rows="25" cols="90"></textarea></div>
      <div><input type="submit" value="Get Sentiment Value"></div>
    </form>
    <p>By: 
    Aditya Gaurav, 
    Aman Gupta,
    Ashish Yadav,
    Kartik Singal</p>
  </body>
</html>
"""

UPDATE_HTML="""\
<html>
<head>
    <style type="text/css">
    </style>
</head>
"""

class MainPage(webapp2.RequestHandler):

    def get(self):
        TrainOnPageLoad();
        self.response.write(MAIN_PAGE_FANCY)

class Nlpminor(webapp2.RequestHandler):

    def post(self):
        output=classify(self.request.get('content'));
        self.response.write(UPDATE_HTML)
        self.response.write('<body><pre><h1>Results : ')
        self.response.write('</h1></pre>')
        if output[0]=='pos':
            self.response.write('<h1><font color="green"> Positive Review </font></h1>')
            self.response.write('<h1>Rating is : <font color="blue"> ' + str(output[1]) + ' </font></h1>')
            t=int(output[1]);
            while t>0:
                self.response.write('<img src="/images/star.png" width="50px" height="50px">');
                t=t-1;

        else:
            self.response.write('<h1><font color="red"> Negative review </font></h1>')
            self.response.write('<h1>Rating is : <font color="blue"> ' + str(output[1]) + ' </font></h1>')
            t=int(output[1]);
            while t>0:
                self.response.write('<img src="/images/star.png" width="50px" height="50px">');
                t=t-1;

        self.response.write('</body></html>')


application = webapp2.WSGIApplication([
    ('/', MainPage),
    ('/sign', Nlpminor),
], debug=True)
