import re, nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk import word_tokenize
from nltk.corpus import stopwords as sw

class MessageWordCloud:
    _contents = ""
    _word_frequency = {}
    # add more class attributes if required

    def __init__(self, content):
        self._contents = content

    # preprocessing the contents of wiki page of the _word, pre
    def preprocess(self, punctuations=False, decapitalisation=False, stopwords=False):
        # remove punctuations
        contents = self._contents
        if punctuations:
            contents = re.sub(r'[^\w\s]', '', self._contents)
            
        # remove stop words
        tokens = word_tokenize(contents)
        if stopwords:
            tokens = [w for w in tokens if not w in sw.words()]

        # decapitalize all tokens
        if decapitalisation:
            tokens = [token.lower() for token in tokens]
        
        # word frequency
        word_dict = {}
        for token in tokens:
            if token in word_dict:
                word_dict[token] += 1
            else:
                word_dict[token] = 1
        self._word_frequency = word_dict

 
    # displaying word cloud
    def create_word_cloud_img(self):
        wordcloud = WordCloud().generate_from_frequencies(self._word_frequency)
        #plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.savefig('word_cloud.png')
        #plt.show()


        

