from logging import info
from string import punctuation
from typing import Tuple

import nltk
from constants import Chunk, Object, Subject, Verb, WordFrequencies
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

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


def get_word_frequencies(chapter_chunk: Chunk) -> WordFrequencies:
    text = " ".join(chapter_chunk)
    tokens = word_tokenize(text)
    tokens_pos = nltk.pos_tag(tokens)
    word_frequencies: WordFrequencies = {}
    for word, pos in tokens_pos:
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
    return get_summary_words(get_word_frequencies(chunk))
