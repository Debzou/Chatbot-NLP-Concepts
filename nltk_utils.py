import nltk
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
    """
    pass


# test function 
a = "How long does shipping take?"
print(a)
a = tokenization(a)
print(a)

words = ['Organize','organizes','organizing','penis']
stemmed_word = [stemming(w) for w in words]
print(stemmed_word)