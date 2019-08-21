import re, nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
from nltk import word_tokenize
from nltk.corpus import stopwords as sw


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

