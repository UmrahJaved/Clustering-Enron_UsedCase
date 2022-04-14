"""
Module for iterating LDA calculation
The result of the iteration is stored to the dataframe as a binary tree list 
The iteration is always for two topics. If the email belongs to the topic 0, there will be 0 
stored to hierarchy information. Otherwise, there will be 1.
"""
from git import Object
import pandas as pd
import pickle
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
import pyLDAvis
import pyLDAvis.gensim_models  # the module 'gensim' has renamed to gensim_models
import dill
from typing import Union
import warnings



#Dropbox account: umrahjaved@yahoo.com
#password: youcanuse

MALLET_PATH = './mallet-2.0.8/bin/mallet'

def load_files(test : bool = True) ->  Union[list, list, Object]:
    """
    Load the files from the dataframe
    param 
    :test: True if the test data is to be loaded
    :list: lemmatized data
    :list: enron emails body as a list
    :Object: corpora module
    """
    with open("./calc_data/data_lemmatized.pkl", 'rb') as my_pickle:
        data_lemmatized = pickle.load(my_pickle)

    with open("./calc_data/data.pkl", 'rb') as my_pickle:
        data = pickle.load(my_pickle)

    with open("./calc_data/enron_df.pkl", 'rb') as my_pickle:
        enron = pickle.load(my_pickle)
    
    #corpora is a dill file (module cannot be pickled)
    with open("./calc_data/corpora.dill", 'rb') as f:
        corpora = dill.load(f)
    
    if test:
        data_lemmatized = data_lemmatized[:1000]
        data = data[:1000]
        enron = enron[:1000]

    return data_lemmatized, data, corpora


def create_id2word(data_lemmatized : list) -> gensim.corpora.Dictionary:
    """
    Create a dictionary from the lemmatized data
    param
    :list: lemmatized data
    :gensim.corpora.Dictionary: dictionary
    """
    id2word = corpora.Dictionary(data_lemmatized)
    #dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
    return id2word

def calc_term_doc_frequency(id2word : gensim.corpora.Dictionary, data_lemmatized : list) -> gensim.corpora.Dictionary:
    """
    Calculate the term-document frequency
    param
    :gensim.corpora.Dictionary: dictionary
    :list: lemmatized data
    :gensim.corpora.Dictionary: term-document frequency
    """
    texts = data_lemmatized #keep track for the previous code, this is not needed
    corpus = [id2word.doc2bow(text) for text in texts]
    return corpus


def calculate_lda_model(corpus : gensim.corpora.Dictionary, id2word : gensim.corpora.Dictionary, num_topics : int = 2) -> gensim.models.ldamodel.LdaModel:
    """
    Build the LDA model
    param
    :gensim.corpora.Dictionary: dictionary
    :gensim.corpora.Dictionary: term-document frequency
    :int: number of topics
    :gensim.models.ldamodel.LdaModel: LDA model
    """
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                id2word=id2word,
                                                num_topics=num_topics,
                                                random_state=100,
                                                update_every=1,
                                                chunksize=100,
                                                passes=10,
                                                alpha='auto',
                                                per_word_topics=True)
    return lda_model

def calculate_score(corpus, lda_model, id2word):
    perplexity = lda_model.log_perplexity(corpus) # a measure of how good the model is. lower the better.
    print('\nPerplexity: ', perplexity)

    warnings.filterwarnings("ignore",category=DeprecationWarning)
    coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print('\nCoherence Score: ', coherence_lda)


def visualize_topic(lda_model : gensim.models.ldamodel.LdaModel, corpus : list, id2word : dict) -> pyLDAvis._prepare.PreparedData:
    """
    Visualize the topics using pyLDAvis
    param:
    :gensim.models.ldamodel.LdaModel: LDA model
    :list: corpus
    :gensim.corpora.Dictionary: dictionary
    :return pyLDAvis._prepare.PreparedData: prepared data
    """
    pyLDAvis.enable_notebook(sort=True)
    vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
    pyLDAvis.display(vis)
    return vis

    
if __name__ == '__main__':
    data_lemmatized, data, corpora = load_files(test=True)
    texts = data_lemmatized
    print(data_lemmatized[1])
    print("Files loaded")
    id2word = create_id2word(data_lemmatized)

    print("id2word created")
    corpus = calc_term_doc_frequency(id2word, data_lemmatized)
    print("corpus created")
    lda_model = calculate_lda_model(corpus, id2word, 2)
    print("lda model calculated")
    calculate_score(corpus, lda_model, id2word)
    print("score calculated")
    #visualize_topic(lda_model, corpus, id2word)
    print("visualized")
    print("Done")


