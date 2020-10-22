import nltk
import numpy as np
#nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
# All model of nltk are already trained 

stemmer = PorterStemmer()

def tokenization(sentence):
    """
    sentence : String
    splite the sentence into a word table
    """
    return nltk.word_tokenize(sentence)

def stemming(word):
    """
    word : String
    group by meaning
    """
    return stemmer.stem(word.lower())

def matrice_of_word(tokenization_sentence,all_words):
    """
    tokenization_sentence = ["hy","who","are","you"]
    all_words = ["bonjour","hy","I","who","are","you"]
    binList = ["0","1","0","1","1","1"]
    """
    tokenization_sentence = [stemming(w) for w in tokenization_sentence]
    binList  = np.zeros(len(all_words))
    for id, word in enumerate(all_words):
        if word in tokenization_sentence:
            binList[id] = 1.0
    return binList

   


# test function 
# a = "How long does shipping take?"
# print(a)
# a = tokenization(a)
# print(a)

# words = ['Organize','organizes','organizing','penis']
# stemmed_word = [stemming(w) for w in words]
# print(stemmed_word)

# sentence = ["hy","who","are","you"]
# words = ["bonjour","hy","I","who","are","you"]
# print(matrice_of_word(sentence,words))