#!/usr/bin/env python
# -*- coding: utf-8 -*-

from scipy import sparse
from scipy import linalg
from scipy import matrix
# from scipy.sparse.linalg.eigen.arpack import *

import math
import sys
import pickle
import numpy

if (len(sys.argv) != 4):
    print("usage: python do-svd.py num-eig-values min-df path-of-doc.tsv")
    sys.exit()

# numEig = 120
# minDF = 10

numEig = int(sys.argv[1])
minDF = int(sys.argv[2])

path = sys.argv[3]

wordToWordID = dict()
wordIDToWord = list()

tweetIDToDocID = dict()
docID = 0
wordID = 0

stream = open(path, "r")
for line in stream:
    line = line[:-1]
    splitted = line.split("\t")
    if len(splitted) < minDF:
        continue
    
    
    word = splitted[0]
    n = len(splitted)

    wordToWordID[word] = wordID
    wordIDToWord.append(word)
    
    i = 0
    for element in splitted:
        if i == 0:
            i = i + 1
            continue
        
        tweetID = element
        
        # if tweetIDToDocID.has_key(tweetID) == False:
        if (tweetID in tweetIDToDocID) == False:
            tweetIDToDocID[tweetID] = docID
            docID = docID + 1
            
        targetDocID = tweetIDToDocID[tweetID]
            
    i = i + 1
    wordID = wordID + 1
            
maxWordID = wordID
maxDocID = docID
            
print(maxWordID)
print(maxDocID)

stream.close

bMat = sparse.lil_matrix((maxWordID + maxDocID + 1, maxWordID + maxDocID + 1))

wordID = 0
stream = open(path, "r")
for line in stream:
    line = line[:-1]
    splitted = line.split("\t")
    
    if len(splitted) < minDF:
        continue
    
    df = int(len(splitted))
    # idf = 1.0 / math.log10(df)
    # idf = 1.0
    idf = math.log10((maxDocID + 1) / df)

    word = splitted[0]
    n = len(splitted)
    
    i = 0
    for element in splitted:
        if i == 0:
            i = i + 1
            continue
        
        tweetID = element
        
        targetDocID = tweetIDToDocID[tweetID]

        if wordID > maxWordID:
            continue

        if targetDocID > maxDocID:
            continue

        # print str(wordID) + " : " + str(targetDocID) + " : " + str(maxDocID + targetDocID) + " : " + str(maxWordID) + " : " + str(maxDocID)
        
        bMat[wordID, maxWordID + targetDocID] = bMat[wordID, maxWordID + targetDocID] + 1.0 * idf
        bMat[maxWordID + targetDocID, wordID] = bMat[maxWordID + targetDocID, wordID] + 1.0 * idf
            
    i = i + 1
    wordID = wordID + 1

print("matrix construction end")

eval, evec = sparse.linalg.eigsh(bMat, numEig)

print("eig is calced")

print(type(eval))
print(type(evec))

# uMat = sparse.lil_matrix((maxWordID, numEig))
# uMatT = sparse.lil_matrix((numEig, maxWordID))
uMat = numpy.ndarray(shape=(maxWordID, numEig), dtype=float)
uMatT = numpy.ndarray(shape=(numEig, maxWordID), dtype=float)

for i in range(maxWordID):
    for j in range(numEig):
        uMat[i, j] = evec[i, j]
        uMatT[j, i] = evec[i, j]

vMat = numpy.ndarray(shape=(maxDocID, numEig), dtype=float)
vMatT = numpy.ndarray(shape=(numEig, maxDocID), dtype=float)

for i in range(maxDocID + maxWordID):
    if i < maxWordID:
        continue
    
    for j in range(numEig):
        vMat[i - maxWordID, j] = evec[i, j]
        vMatT[j, i - maxWordID] = evec[i, j]


# sigMat = sparse.lil_matrix((numEig, numEig))
sigMat = numpy.ndarray(shape=(numEig, numEig), dtype=float)
for i in range(numEig):
    for j in range(numEig):
        
        if i == j:
            sigMat[i, i] = eval[i]
        else:
            sigMat[i, j] = 0.0

print("sig mat is made")

pickle.dump(uMat, open("uMat.p", "wb"))
pickle.dump(uMatT, open("uMatT.p", "wb"))
# pickle.dump(vMat, open("vMat.p", "wb"))
# pickle.dump(vMatT, open("vMatT.p", "w"))
pickle.dump(sigMat, open("sigMat.p", "wb"))
pickle.dump(wordIDToWord, open("wordIDToWord.p", "wb"))
pickle.dump(wordToWordID, open("wordToWordID.p", "wb"))

