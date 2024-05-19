"""


"""

import re
from gensim.models import Word2Vec, LsiModel
import gensim
from gensim import corpora
from load import Loader
from training import Trainer
import nltk
import spacy
from tqdm import tqdm


from utils import embedding_info

# Training the dailydialog
loader = Loader()
daily_dialog = loader.load_txt("./data/dailydialog/dialogues_text.txt")
def preprocess_dailydialog(corpus):
    """
    Preprocess the Daily Dialog Corpus
    """
    preprocessed_scentences = []
    for dialogue in corpus:
        sentences = dialogue.split("__eou__")
        
        # Tokenize each sentence and remove empty sentences
        tokenized_sentences = [re.findall(r'\w+', sentence.lower()) for sentence in sentences if sentence.strip()]
        preprocessed_scentences.extend(tokenized_sentences)
    return preprocessed_scentences
preprocessed_dailydialog = preprocess_dailydialog(daily_dialog)
print("Total scentences", len(preprocessed_dailydialog))

# Train Word2Vec Model
vector_size = 100
window = 5
min_count = 1
trainer = Trainer(preprocessed_dailydialog, "dailydialog", "word2vec", vector_size, window, min_count, 4)
model = trainer.train()

######################################################################

# Training the text8 with Word2Vec
# loader = Loader()
# text8_corpus = loader.load_txt("./data/text8.txt")
# def preprocess_text8(corpus):
#     k = 1000
#     corpus = corpus.split()
#     corpus = [corpus[i:i+k] for i in range(0, len(corpus), k)]
#     return corpus
# text8_corpus = preprocess_text8(text8_corpus[0])
# print("Total sentences", len(text8_corpus))
# trainer = Trainer(text8_corpus, "text8", "LsiModel", 100, 5, 1, 4)
# model = trainer.train()

######################################################################

# 20 News Group Json
# https://raw.githubusercontent.com/selva86/datasets/master/newsgroups.json
# https://blog.csdn.net/lwhsyit/article/details/82750218
loader = Loader()
newsgroups = loader.load_json("./data/20newsgroups.json")

def preprocess_20newsgroup(corpus):
    # convert to json to list
    content = corpus["content"]
    corpus = [content[key] for key in content.keys()]  
    print("Corpus length", len(corpus)) 
     
    corpus = [re.sub(r'\S*@\S*\s?', '', sent) for sent in corpus] # remove emails
    corpus = [re.sub(r'\s+', ' ', sent) for sent in corpus] # remove newline chars
    corpus = [re.sub(r"\'", "", sent) for sent in corpus]     # remove distracting single quotes
    
    # Tokenize each sentence into words, remove punctuations and uncessary characters
    tokenized_sentences = [re.findall(r'\w+', sentence.lower()) for sentence in corpus]
    
    # Remove stopwords
    nltk.download('stopwords', download_dir='./data')
    stop_words = open('./data/corpora/stopwords/english').read().splitlines()
    stop_words.extend(['from', 'subject', 're', 'edu', 'use'])
    tokenized_sentences = [[word for word in sentence if word not in stop_words] for sentence in tokenized_sentences]
    
    # Bigrams and Trigrams
    bigram = gensim.models.Phrases(tokenized_sentences, min_count=5, threshold=100)
    # trigram = gensim.models.Phrases(bigram[tokenized_sentences], threshold=100)
    
    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    # trigram_mod = gensim.models.phrases.Phraser(trigram)
    
    # Make bigrams
    bigram_sentences = []
    for sentence in tqdm(tokenized_sentences):
        bigram_sentences.append(bigram_mod[sentence])
    
    # Lemmatization - only keep nouns, adjectives, verbs and adverbs
    allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']
    lemmitized_sentences = []
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    for text in tqdm(bigram_sentences):
        text = nlp(" ".join(text))
        lemmitized_sentences.append([token.lemma_ for token in text if token.pos_ in allowed_postags])

    # # See trigram example
    # # print(trigram_mod[bigram_mod[tokenized_sentences[0]]])
    # return lemmitized_sentences
    
    return  lemmitized_sentences
    
preprocessed_20newsgroup = preprocess_20newsgroup(newsgroups)
print(preprocessed_20newsgroup[0])
trainer = Trainer(preprocessed_20newsgroup, 
                corpus_name="20newsgroup", 
                embedding_method="LdaModel", 
                num_topics=20,
                window=5, 
                min_count=1, 
                workers=4,
                save_corpus=True
)
model = trainer.train()

######################################################################

# Run Classification Evaluation on GoogleNews pretrained embeddings
