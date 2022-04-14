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

from regex import D


#Dropbox account: umrahjaved@yahoo.com
#password: youcanuse

MALLET_PATH = './mallet-2.0.8/bin/mallet'

def load_files(test : bool = True) ->  Union[list, pd.DataFrame]:
    """
    Load the files from the dataframe
    param 
    :test: True if the test data is to be loaded
    return:
    :list: list of lemmatized emails
    :pd.DataFrame: enron original email dataframe
    """
    # We work with these. These are the one's that define what data we are using
    with open("./calc_data/data_lemmatized.pkl", 'rb') as my_pickle:
        data_lemmatized = pickle.load(my_pickle)

    # This is needed when result is made
    with open("./calc_data/enron_df.pkl", 'rb') as my_pickle:
        enron = pickle.load(my_pickle)

    if test:
        data_lemmatized = data_lemmatized[:1000]
        #data = data[:1000]
        enron = enron[:1000]
    return data_lemmatized, enron


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

def calculate_score(corpus, lda_model, id2word, data_lemmatized):
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

def categorise_emails(ldamodel : gensim.models.ldamodel.LdaModel, corpus : list, texts : list) -> pd.DataFrame:
    """
    Creates a dataframe that contains information what is the dominant topic to this email. Also how
    much of the email is in the topic. 
    param:
    :gensim.models.ldamodel.LdaModel: LDA model
    :list: corpus
    :list: texts (lemmatized emails)
    """
    sent_topics_df = pd.DataFrame()
    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(
            row[0], key=lambda x: (x[1]), 
            reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output NOT NEEDED
    #contents = pd.Series(texts)
    #sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)

def divide_df(cat_df :pd.DataFrame) -> Union [pd.DataFrame, pd.DataFrame]:
    """
    Divide the dataframe into two dataframes, one for the emails that are in the topic 0 and one for the emails that are in the
    topic 1.
    param:
    :pd.DataFrame: dataframe
    :return: pd.DataFrame, pd.DataFrame
    """
    cat_df_topic = cat_df[cat_df['Dominant_Topic'] == 0]
    cat_df_not_topic = cat_df[cat_df['Dominant_Topic'] != 0]
    return cat_df_topic, cat_df_not_topic

def sub_level():
    #testing in Jupyter how this should be done efficiently
    pass

def add_enron_columns(cat_emails : pd.DataFrame, enron : pd.DataFrame) -> pd.DataFrame:
    cat_emails['ID'] = enron['Message-ID']
    cat_emails['Subject'] = enron['Subject']
    cat_emails['enron_content'] = enron['content']
    cat_emails['From'] = enron['From']
    cat_emails['To'] = enron['To']
    cat_emails['Cc'] = enron['X-cc']
    cat_emails['Bcc'] = enron['X-bcc']


def first_level():
    data_lemmatized, enron = load_files(test=True)
    id2word = create_id2word(data_lemmatized)
    corpus = calc_term_doc_frequency(id2word, data_lemmatized)
    lda_model = calculate_lda_model(corpus, id2word, 2)
    categorised_emails = categorise_emails(lda_model, corpus, data_lemmatized)
    categorised_emails = add_enron_columns(categorised_emails, enron)



    #calculate_score(corpus, lda_model, id2word, data_lemmatized)




    
if __name__ == '__main__':    
    first_level()



