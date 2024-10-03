import nltk
nltk.download('stopwords')
nltk.download('punkt')

def tokenize(text: str) -> list[str]:
    return nltk.word_tokenize(text)

def remove_stopwords(tokens: list[str]) -> list[str]:
    return [token for token in tokens if token not in nltk.corpus.stopwords.words('russian')]

def stem(tokens: list[str]) -> list[str]:
    return [nltk.stem.snowball.SnowballStemmer('russian').stem(token) for token in tokens]

def text_preprocess(text: str) -> list[str]:
    text = text.lower()
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = stem(tokens)
    return tokens