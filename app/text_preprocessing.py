import re, nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
from nltk import word_tokenize
from nltk.corpus import stopwords as sw


# preprocessing the contents of wiki page of the _word, pre
def preprocess(contents='', punctuations=False, decapitalisation=False, stopwords=False):
    '''
    
    '''
    
    #init
    word_frequency={}
    # Normalize text
    ##lower case, remove non -alphabet, numeric
    contents = re.sub(r"[^a-zA-Z0-9]", " ", contents.lower())

    # remove punctuations
    if punctuations:
        contents = re.sub(r'[^\w\s]', '', contents)

    # remove stop words
    tokens = word_tokenize(contents)
    cachedStopWords = set(sw.words("english")) #dramatic boost from not having to read the list    from disk every time,, turn to set make another boost
    
    if stopwords:
        tokens = [w for w in tokens if w not in cachedStopWords]
    
    # word frequency
    word_dict = {}
    for token in tokens:
        if token in word_dict:
            word_dict[token] += 1
        else:
            word_dict[token] = 1
    word_frequency = word_dict

    return word_frequency #TODO: save word_frequency in a file and load to plot
