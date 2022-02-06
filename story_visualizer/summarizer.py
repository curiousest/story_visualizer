import itertools
from logging import info
from string import punctuation
from typing import List, Tuple

import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from story_visualizer.constants import (
    Chunk,
    Object,
    Subject,
    TokenizedText,
    Verb,
    WordFrequencies,
)

wordnet_lemmatizer = WordNetLemmatizer()
nltk.download("omw-1.4")
nltk.download("wordnet")
wordnet_map = {"N": wordnet.NOUN, "V": wordnet.VERB, "J": wordnet.ADJ, "R": wordnet.ADV}

nltk.download("stopwords")
stop_words = stopwords.words("english")
punctuation = punctuation + "\n" + "\r" + "`"


def include_word(word):
    if word.lower() in stop_words:
        return False
    if word.lower() in punctuation:
        return False
    if not str.isalpha(word[0]):
        return False
    return True


def get_word_frequencies(tokenized_text: TokenizedText) -> WordFrequencies:
    word_frequencies: WordFrequencies = {}
    for word, pos in tokenized_text:
        if include_word(word):
            if (word, pos[0]) not in word_frequencies.keys():
                # pos[0] condenses parts of speech
                # https://stackoverflow.com/questions/15388831/what-are-all-possible-pos-tags-of-nltk
                word_frequencies[(word, pos[0])] = 1
            else:
                word_frequencies[(word, pos[0])] += 1
    return word_frequencies


def get_summary_words(
    word_frequencies: WordFrequencies,
) -> Tuple[Subject, Verb, Object, Object]:
    top_frequencies = sorted(word_frequencies.items(), key=lambda a: a[1], reverse=True)
    nouns = filter(lambda f: f[0][1] == "N", top_frequencies)
    verbs = filter(lambda f: f[0][1] == "V", top_frequencies)
    subject = next(nouns)[0][0]
    verb = next(verbs)[0][0]
    object1 = next(nouns)[0][0]
    object2 = next(nouns)[0][0]
    return subject, verb, object1, object2


def get_summary(chunk: Chunk) -> Tuple[Subject, Verb, Object, Object]:
    processed_chunk: List[TokenizedText] = [
        preprocess_text(paragraph) for paragraph in chunk
    ]
    word_frequencies = get_word_frequencies(itertools.chain(*processed_chunk))
    return get_summary_words(word_frequencies)


def lemmatize(word: str, pos: str):
    pos = wordnet_map.get(pos[0], None)
    return (
        wordnet_lemmatizer.lemmatize(word, pos=pos)
        if pos
        else wordnet_lemmatizer.lemmatize(word)
    )


def preprocess_text(text: str) -> TokenizedText:
    tokens: List[str] = word_tokenize(text.lower())
    tokens_pos: TokenizedText = nltk.pos_tag(tokens)
    lemmatized_text: TokenizedText = [
        (lemmatize(word, pos), pos) for (word, pos) in tokens_pos
    ]
    return lemmatized_text
