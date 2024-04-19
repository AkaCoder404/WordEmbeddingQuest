"""
Title: Utils

"""



import bz2
import gzip
from os import path
import tarfile
import io
from itertools import islice, chain
from six import string_types, text_type
import numpy as np

from gensim.models import Word2Vec

def any2utf8(text, errors='strict', encoding='utf8'):
    """Convert a string (unicode or bytestring in `encoding`), to bytestring in utf8."""
    if isinstance(text, text_type):
        return text.encode('utf8')
    # do bytestring -> unicode -> utf8 full circle, to ensure valid utf8
    return text_type(text, encoding, errors=errors).encode('utf8')

to_utf8 = any2utf8

# Works just as good with unicode chars
_delchars = [chr(c) for c in range(256)]
_delchars = [x for x in _delchars if not x.isalnum()]
_delchars.remove('\t')
_delchars.remove(' ')
_delchars.remove('-')
_delchars.remove('_')  # for instance phrases are joined in word2vec used this char
_delchars = ''.join(_delchars)
_delchars_table = dict((ord(char), None) for char in _delchars)

def standardize_string(s, clean_words=True, lower=True, language="english"):
    """
    Ensures common convention across code. Converts to utf-8 and removes non-alphanumeric characters

    Parameters
    ----------
    language: only "english" is now supported. If "english" will remove non-alphanumeric characters

    lower: if True will lower strńing.

    clean_words: if True will remove non alphanumeric characters (for instance '$', '#' or 'ł')

    Returns
    -------
    string: processed string
    """

    assert isinstance(s, string_types)

    if not isinstance(s, text_type):
        s = text_type(s, "utf-8")

    if language == "english":
        s = (s.lower() if lower else s)
        s = (s.translate(_delchars_table) if clean_words else s)
        return s
    else:
        raise NotImplementedError("Not implemented standarization for other languages")


def batched(iterable, size):
    sourceiter = iter(iterable)
    while True:
        batchiter = islice(sourceiter, size)
        try:
            yield chain([next(batchiter)], batchiter)
        except StopIteration:
            return


def _open(file_, mode='r'):
    """Open file object given filenames, open files or even archives."""
    if isinstance(file_, string_types):
        _, ext = path.splitext(file_)
        if ext in {'.gz'}:
            if mode == "r" or mode == "rb":
                # gzip is extremely slow
                return io.BufferedReader(gzip.GzipFile(file_, mode=mode))
            else:
                return gzip.GzipFile(file_, mode=mode)
        if ext in {'.bz2'}:
            return bz2.BZ2File(file_, mode=mode)
        else:
            return io.open(file_, mode, **({"encoding": "utf-8"} if "b" not in mode else {}))
    return file_


def embedding_info(embeddings: Word2Vec):
    print("Vector size:", embeddings.vector_size)
    print("Dictionary size", len(embeddings.wv.index_to_key))
    print("Window size", embeddings.window)
    print("Total training time", embeddings.total_train_time)
    

def load_word_vectors(file_path):
    """
    Load embeddings with the following format:
    <word> <vector>
    """
    word_vectors = {}
    with open(file_path, 'r') as file:
        for line in file:
            word, *vector = line.split()
            word_vectors[word] = np.array(vector, dtype=np.float32)
    return word_vectors

def most_similar_words(embeddings, word, k):
    """
    Get the top k most similar words to a word
    embeddings is of form {word1: vector1, word2: vector2, ...}
    """
    
    word = standardize_string(word)
    word_vector = embeddings[word]
    
    similarity = {}
    for w, v in embeddings.items():
        try:
            similarity[w] = np.dot(word_vector, v) / (np.linalg.norm(word_vector) * np.linalg.norm(v))
        except Exception as e:
            continue

        
    # Return list of [(word, similarity), ...]
    return sorted(similarity.items(), key=lambda x: x[1], reverse=True)[:k]
        
    # return sorted(similarity, key=lambda x: similarity[x], reverse=True)[:k]
