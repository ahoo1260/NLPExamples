
import pickle
from collections import Counter
import json
from nltk import ngrams
from source.util.cleaning import clean_contractions, clean_stopwords, clean_special_chars, cleanhtml,stemming_word
import os
import pandas as pd
def cleanData(data):
    data = data.apply(lambda x: x.lower())
    data = data.apply(lambda x: cleanhtml(x))
    data = data.apply(lambda x: clean_stopwords(x))
    # data=data.apply(lambda x: stemming_word(x))
    data = data.apply(lambda x: clean_special_chars(x))
    data = data.apply(lambda x: clean_contractions(x))
    clean_data=data
    return clean_data

def writeListInFile(myArray, fileName):
    with open(fileName, "wb") as fp:
        pickle.dump(myArray, fp)


def readFileIntoList(filename):
    with open(filename, "rb") as fp:
        myArray=pickle.load(fp)
    return myArray

def convertListToString(data):
    return ''.join(data)

def get_k_MostFrequentWordsFromText(data,k):
    data=convertListToString(data)
    split_it = data.split()
    counter = Counter(split_it)
    most_occur = counter.most_common(k)

    print(most_occur)
    return most_occur
def get_k_MostFrequentNgrams(data,most_frequent_K, N_gram):
    ngram_counts=getNgrams(data,N_gram)
    print("number of total " + str(N_gram)+" is:"+str(len(ngram_counts)))
    most_occur=ngram_counts.most_common(most_frequent_K)
    # most_occur_without_accurance_number = [a for a, b in most_occur]
    return most_occur

def getNgrams(data,N):
    data = convertListToString(data)
    ngram_counts = Counter(ngrams(data.split(), N))
    return ngram_counts

def readAnswersFromJson(jsonFile, number_of_questions):
    persons=[]
    person=[]
    with open(jsonFile) as json_file:
        data = json.load(json_file)

    s=int(len(data)/number_of_questions)

    for i in range(0,s):
        for j in range(1,number_of_questions+1):
            person.append(data[str(j+i*number_of_questions)])
        persons.append(person)
        person=[]
    return persons









