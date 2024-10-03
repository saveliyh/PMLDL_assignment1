import nltk
import string
import re

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    
    nltk.download('stopwords')

def remove_punctuation(text: str) -> str:
    text = text.translate(str.maketrans('', '', string.punctuation))    
    return text

def remove_not_space(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    while '  ' in text:
        text = text.replace('  ', ' ')
    return text


def tokenize(text: str) -> list[str]:
    return nltk.word_tokenize(text)

def remove_stopwords(tokens: list[str]) -> list[str]:
    return [token for token in tokens if token not in nltk.corpus.stopwords.words('russian')]

def stem(tokens: list[str]) -> list[str]:
    return [nltk.stem.snowball.SnowballStemmer('russian').stem(token) for token in tokens]



def text_preprocess(text: str) -> list[str]:
    text = text.lower()
    text = remove_punctuation(text)
    text = remove_not_space(text)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = stem(tokens)
    return tokens