"""

"""

from gensim.models import Word2Vec, FastText, LsiModel, LdaModel
from gensim.models import KeyedVectors
from gensim.models.coherencemodel import CoherenceModel
from gensim import corpora
from time import time
import json


class Trainer:
    def __init__(self, corpus, 
                 corpus_name, 
                 embedding_method, 
                 window, 
                 min_count, 
                 workers,
                 vector_size=100,
                 num_topics=20,
                 save=True,
                 save_corpus=False
        ):
        """
        
        @param corpus: The corpus to train the model on (preprocessed)
        @param corpus_name: The name of the corpus
        @param embedding_method: The embedding method to use, i.e Word2Vec, FastText, etc.
        @param vector_size: The size of the vector
        @param window: The window size
        @param min_count: The minimum count
        @param workers: The number of workers
        @param save_path: The path to save the model
        @param save: If True, save the model
        """
        self.corpus = corpus
        self.corpus_name = corpus_name
        self.embedding_method = embedding_method
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.save_path = f"./out/{corpus_name}_{embedding_method}_{vector_size}.model"
        self.save = save
        self.num_topics = num_topics
        self.save_corpus = save_corpus
        
        
    def daily_dialog_word2vec(self):
        """
        
        """
        
    def daily_dialog_glove(self):
        """
        
        
        """
        
        
    def max_coherence_value(self, dictionary, doc_term_matrix, doc_clean, stop, start=2, step=3):
        """
        Compute c_v coherence for various number of topics to find the optimal number of topics
        Input   : dictionary : Gensim dictionary
                corpus : Gensim corpus
                texts : List of input texts
                stop : Max num of topics
        purpose : Compute c_v coherence for various number of topics
        Output  : model_list : List of LSA topic models
                coherence_values : Coherence values corresponding to the LDA model with respective number of topics
        """
        coherence_values = []
        for num_topics in range(start, stop, step):
            print(num_topics)
            # generate LSA model
            model = LsiModel(doc_term_matrix, num_topics=num_topics, id2word = dictionary)  # train model
            coherencemodel = CoherenceModel(model=model, texts=doc_clean, dictionary=dictionary, coherence='c_v')
            coherence_values.append(coherencemodel.get_coherence())
            
        return coherence_values.index(max(coherence_values)), coherence_values
        
        
    def train(self):
        """
        Train the model
        """
        print(f"Training {self.embedding_method} model on {self.corpus_name} corpus")
        
        assert self.embedding_method in ["Word2Vec", "FastText", "GloVe", "LdaModel", "LsiModel"], "Invalid embedding method"
        
        if self.embedding_method == "Word2Vec":
            model = Word2Vec(self.corpus, 
                             vector_size=self.vector_size, 
                             window=self.window, 
                             min_count=self.min_count, 
                             workers=self.workers)
            
        
        elif self.embedding_method == "FastText":
            model = FastText(self.corpus, 
                             vector_size=self.vector_size, 
                             window=self.window, 
                             min_count=self.min_count, 
                             workers=self.workers)
            
        elif self.embedding_method == "GloVe":
            pass
        
        elif self.embedding_method == "LdaModel":
            dictionary = corpora.Dictionary(self.corpus)
            doc_matrix = [dictionary.doc2bow(doc) for doc in self.corpus]
            model = LdaModel(corpus=doc_matrix,
                             id2word=dictionary,
                             num_topics=self.num_topics,
                             random_state=100,
                                update_every=1,
                                chunksize=100,
                                passes=10,
                                alpha='auto',
                                per_word_topics=True
            )
            
            self.save_path = f"./out/{self.corpus_name}_{self.embedding_method}_{self.num_topics}.model"
                                     
        elif self.embedding_method == "LsiModel":
            print("Getting optimal number of topics")
            dictionary = corpora.Dictionary(self.corpus)
            doc_matrix = [dictionary.doc2bow(doc) for doc in self.corpus]
            max_coherence, _ = self.max_coherence_value(dictionary, doc_matrix, self.corpus, 20)

            
            print(f"Optimal number of topics: {max_coherence}")
            model = LsiModel(doc_matrix,
                             id2word=dictionary,
                             num_topics=max_coherence
            )
            self.save_path = f"./out/{self.corpus_name}_{self.embedding_method}_{max_coherence}.model"
        
        try: 
            print(f"Training completed in {model.total_train_time} seconds")
        except AttributeError:
            """"""
        
        if self.save_corpus:
            """ 
            Corpus is a list of lists, each list is a sentence
            Best way to save is with a json file
            """
            json.dump(self.corpus, open(f"./out/{self.corpus_name}_corpus.json", "w"))
            
        
        if self.save:
            model.save(self.save_path, )
            print(f"Model saved at {self.save_path}")
        

        return model
        
    
    