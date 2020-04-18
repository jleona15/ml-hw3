from __future__ import print_function
from nltk.stem import *
from nltk.stem.porter import *

import sys
import os
import math
import string

stem_stopwords = False

learning_rate = 0.001
training_iterations = 3

def stem(text):
    
    l = list(text)

    stemmer = PorterStemmer()

    for p in string.punctuation:
        while True:
            try:
                l[l.index(p)] = ' '
            except:
                break

    word_list = "".join(l).split()

    if stem_stopwords:
        try:
            stopwords = open("stopwords.txt").read().split()
        except:
            print("Error: stopwords.txt not found, please add a stopwords file")
            exit()
        for s in stopwords:
            while True:
                try:
                    word_list.remove(s)
                except:
                    break

    return [stemmer.stem(plural).lower() for plural in word_list]


def train():
    v = get_vocabulary()

    m = list()

    m.append(list())
    m.append(list())

    for _v in v:
        m[0].append(_v)
        m[1].append(0.0)

    m[1].append(0.0)

    for training_iter in range(training_iterations):
        #temp = m.copy()
        for f_name in os.listdir(os.getcwd() + "/train/spam"):
            result = evaluate(m, os.getcwd() + "/train/spam/" + f_name, True, -1)
        #if temp == m:
        #    print("ERR")
            #exit()
        #temp = m.copy()
        for f_name in os.listdir(os.getcwd() + "/train/ham"):
            result = evaluate(m, os.getcwd() + "/train/ham/" + f_name, True, 1)
        #if temp == m:
        #    print("ERR")
            #exit()
    return m
     

def get_vocabulary():
    training_files = os.listdir(os.getcwd() + "/train/spam")
    for i in range(len(training_files)):
        training_files[i] = os.getcwd() + "/train/spam/" + training_files[i]

    training_ham = os.listdir(os.getcwd() + "/train/ham/")
    for i in range(len(training_ham)):
        training_ham[i] = os.getcwd() + "/train/ham/" + training_ham[i]

    training_files.extend(training_ham)

    v_word = list()

    for f_name in training_files:
        #print('a')
        if f_name.split('.')[-1] == "txt":
            #print('b')
            f = open(f_name, "r")
            #print(f_name)
            #print(f.read())
            lines = stem(f.read())
            words = set(lines)
            #print(words)
            for w in words:
                if not (w in v_word):
                    v_word.append(w)

    return v_word
            
def evaluate(m, f_name, train_flag, expected):
    
    #tokens = stem(open(f_name).read())

    text = stem(open(f_name, "r").read()) 

    tc = list()

    for p in string.punctuation:
        while p in text:
            text[text.index(p)] = ' '
 
    text = "".join("".join(text).split())

    summed = 0.0

    for i in range(len(m[0])):
        temp = text.count(m[0][i])
        
        summed = summed + (m[1][i] * temp)
        #if temp > 0.0:
            #print(temp)
        if train_flag:
            tc.append(temp)

    summed = summed + m[1][-1]

    #print("Sum: " + str(summed))

    #print(model[0][0])

    
    #print(str(spam_score) + str(ham_score))

    if train_flag:
        tc.append(1)

    result = 0

    if summed >= 0.0:
        result = 1
    else:
        result = -1

    if train_flag:
        for i in range(len(m[1])):
            #temp = m[1][i]
            m[1][i] = m[1][i] + (learning_rate * tc[i] * (expected - result))
            #if temp - m[1][i] < -0.0001:
            #    print("UP (" + str(temp) + ", " + str(m[1][i]) + ")")
            #elif temp - m[1][i] > 0.0001:
            #    print("DOWN (" + str(temp) + ", " + str(m[1][i]) + ")")
    #else:
        #print(summed)
    return result

if __name__ == "__main__":
    print("Training    Test_with_stopwords    Test_without_stopwords")
    m = train()
    #print(m)
    stem_stopwords = True
    m2 = train()

    count1 = 0
    mismatch1 = 0

    for f_name in os.listdir(os.getcwd() + "/train/spam"):
        count1 += 1
        if evaluate(m, os.getcwd() + "/train/spam/" + f_name, False, 0) == 1:
            mismatch1 += 1

    for f_name in os.listdir(os.getcwd() + "/train/ham"):
        count1 += 1
        if evaluate(m, os.getcwd() + "/train/ham/" + f_name, False, 0) == -1:
            mismatch1 += 1

    
    count2 = 0
    mismatch2 = 0

    for f_name in os.listdir(os.getcwd() + "/test/spam"):
        count2 += 1
        if evaluate(m, os.getcwd() + "/test/spam/" + f_name, False, 0) == 1:
            mismatch2 += 1

    for f_name in os.listdir(os.getcwd() + "/test/ham"):
        count2 += 1
        if evaluate(m, os.getcwd() + "/test/ham/" + f_name, False, 0) == -1:
            mismatch2 += 1
 
    count3 = 0
    mismatch3 = 0

    for f_name in os.listdir(os.getcwd() + "/test/ham"):
        count3 += 1
        if evaluate(m2, os.getcwd() + "/test/ham/" + f_name, False, 0) == -1:
            #print("A")
            mismatch3 += 1

    for f_name in os.listdir(os.getcwd() + "/test/spam"):
        count3 += 1
        if evaluate(m2, os.getcwd() + "/test/spam/" + f_name, False, 0) == 1:
            mismatch3 += 1
            #print("B")

    #print(m[1])

    #for i in m[1]:
        #if i > 0:
            #print(i)

    print("%.2f"%(1 - (mismatch1/count1)) + "        " + "%.2f"%(1 - (mismatch2/count2)) + "                   " + "%.2f"%(1 - (mismatch3/count3))) 

    #print(m)
