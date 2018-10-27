import string
import warnings
from nltk.util import ngrams
warnings.filterwarnings('ignore')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

def cleaning(data):
    #Remove whitespaces
    # remove whitespaces
    data['text'] = data['text'].str.strip()

    # lowercase the text
    data['text'] = data['text'].str.lower()

    # remove punctuation
    punc = string.punctuation
    table = str.maketrans('', '', punc)
    data['text'] = data['text'].apply(lambda x: x.translate(table))

    # tokenizing each message
    data['word_tokens'] = data.apply(lambda x: x['text'].split(' '), axis=1)
    #data['word_tokens'] =

    # removing stopwords
    cachedStopWords = stopwords.words("english")
    cachedStopWords.append('gun')
    cachedStopWords.append('violence')
    cachedStopWords.append('active')
    cachedStopWords.append('shooter')

    data['cleaned_text'] = data.apply(lambda x: [word for word in x['word_tokens'] if word not in cachedStopWords],
                                      axis=1)
    data['word_tokens_bigrams'] =  [list(zip(x,x[1:])) for x in data.cleaned_text.values.tolist()]
    data['word_tokens_bigrams'] = data.apply(lambda  x: (list(' '.join(w) for w in x['word_tokens_bigrams'])), axis=1)

    # stemming
    #TODO: Remove stemmed words
    #ps = PorterStemmer()
    #data['stemmed'] = data.apply(lambda x: [ps.stem(word) for word in x['cleaned_text']], axis=1)

    # remove single letter words
    data['final_text'] = data.apply(lambda x: ' '.join([word for word in x['cleaned_text'] if len(word) > 1]), axis=1)

    # label encoding negative tweets=0 and gun violence tweets=1
    data.loc[data['type'] == 'negative tweets', 'type'] = 0
    data.loc[data['type'] == 'gun violence', 'type'] = 1

    return data