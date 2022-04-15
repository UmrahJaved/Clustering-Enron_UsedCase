"""
Module for iterating LDA calculation
The result of the iteration is stored to the dataframe as a binary tree list
The iteration is always for two topics. If the email belongs to the topic 0, there will be 0
stored to hierarchy information. Otherwise, there will be 1.
"""

from operator import index
import pandas as pd
import pickle
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
import pyLDAvis
import pyLDAvis.gensim_models  # the module 'gensim' has renamed to gensim_models
from typing import Union
import warnings
from pyparsing import col
warnings.simplefilter(action='ignore', category=FutureWarning)

#from regex import D


#Dropbox account: umrahjaved@yahoo.com
#password: youcanuse
#MALLET_PATH = './mallet-2.0.8/bin/mallet'

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
        data_lemmatized = data_lemmatized[:100000]
        #data = data[:1000]
        enron = enron[:100000]
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

def calc_term_doc_frequency(id2word : gensim.corpora.Dictionary,
                            data_lemmatized : list) -> gensim.corpora.Dictionary:
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


def calculate_lda_model(corpus : gensim.corpora.Dictionary,
                        id2word : gensim.corpora.Dictionary,
                        num_topics : int = 2) -> gensim.models.ldamodel.LdaModel:
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
    # a measure of how good the model is. lower the better.
    perplexity = lda_model.log_perplexity(corpus)
    coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized,
                                        dictionary=id2word, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()


def visualize_topic(lda_model : gensim.models.ldamodel.LdaModel,
                    corpus : list, id2word : dict) -> pyLDAvis._prepare.PreparedData:
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


def categorise_emails(ldamodel : gensim.models.ldamodel.LdaModel,
                        corpus : list, texts : list) -> pd.DataFrame:
    """
    Creates a dataframe that contains information what is the dominant
    topic to this email. Also how much of the email is in the topic.
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
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num),
                                    round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output NOT NEEDED
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return sent_topics_df


def convert_topic_and_perc(df : pd.DataFrame) -> pd.DataFrame:
    """
    Converting topic and perc to a list values in the
    dataframe and creating a new column for them.
    param:
    :pd.DataFrame: dataframe that will be changed
    :return: pd.DataFrame after update
    """
    def convert_to_list(value):
        return [value]
    print(" ----------------------- BEFORE convert_topic_and_perc -----------------------")
    df['Topic_hier'] = df['Dominant_Topic'].apply(convert_to_list)
    print(df['Topic_hier'])
    df['Perc_list'] = df['Perc_Contribution'].apply(convert_to_list)
    print(df['Perc_list'])
    print(" -----------------------        END  (before)         -----------------------")
    return df


def update_topic_and_perc(df : pd.DataFrame) -> pd.DataFrame:
    """
    Adding new topic level and perc contribution for each email's list
    param:
    :pd.DataFrame: dataframe that will be changed
    :return: pd.DataFrame after update
    """
    #print("pd dataframe kutsuessa: ", df.head(1))
    print("ORIGINAL COLUMNS:\n", df.columns)


    print(" ----------------------- BEFORE update update_topic_and_perc -----------------------")
    print((df['Topic_hier']))
    print((df['Perc_list']))
    print(" -----------------------           END             -----------------------")

    new_topic = df['Dominant_Topic'].to_list()
    old_topic_list = df['Topic_hier'].to_list()

    for i, old_topic in enumerate(old_topic_list):
        old_topic.append(new_topic[i])
    df['Topic_hier'] = old_topic_list


    new_perc = df['Perc_Contribution'].to_list()
    old_perc_list = df['Perc_list'].to_list()

    for i, old_perc in enumerate(old_perc_list):
        #old_perc.append(round(new_perc[i],4))
        old_perc.append(new_perc[i])
    df['Perc_list'] = old_perc_list


    print(" ----------------------- AFTER update update_topic_and_perc -----------------------")
    print((df['Topic_hier']))
    print((df['Perc_list']))
    print(" -----------------------           END             -----------------------")
    return df


def divide_df(cat_df :pd.DataFrame) -> Union [pd.DataFrame, pd.DataFrame]:
    """
    Divide the dataframe into two dataframes, one for the emails that are in the
    topic 0 and one for the emails that are in the topic 1.
    param:
    :pd.DataFrame: dataframe
    :return: pd.DataFrame, pd.DataFrame
    """
    cat_df_topic = cat_df[cat_df['Dominant_Topic'] == 0]
    cat_df_not_topic = cat_df[cat_df['Dominant_Topic'] != 0]
    return cat_df_topic, cat_df_not_topic

def convert(dataf : pd.DataFrame) -> pd.DataFrame:
    """
    Empty function for testing only
    """
    length = len(dataf)
    original_columns = dataf[['Lemmatized_Text', 'ID','Subject', 'enron_content', 'From', 'To', 'Cc' ,'Bcc', 'Topic_hier', 'Perc_list']]
    #original_columns = dataf[['Lemmatized_Text', 'ID', 'Topic_hier', 'Perc_list', 'enron_content']]
    #original_columns = dataf[['Lemmatized_Text', 'ID', 'Topic_hier', 'Perc_list']] #easier to debug when less data

    data_lemmitized = dataf['Lemmatized_Text'].to_list() #Lemmatized_Text
    id2word =  create_id2word(data_lemmitized)
    corpus = calc_term_doc_frequency(id2word, data_lemmitized)
    lda_model = calculate_lda_model(corpus, id2word, 2)
    # Get the email topics and dominant topic for each email
    categorised_emails = categorise_emails(lda_model, corpus, data_lemmitized)

    # drop indexes, otherwise the index will be used by concat
    original_columns.reset_index(drop=True, inplace=True)
    categorised_emails.reset_index(drop=True, inplace=True)
    categorised_email_concat = pd.concat([original_columns, categorised_emails], axis=1)
    result = update_topic_and_perc(categorised_email_concat)
    return result

def sub_level(my_df : pd.DataFrame, limit: int = 5) -> list:
    """"
    split the dataframe and convert if there are more than limit number of emails
    """
    list_of_df = []
    df_1, df_2 = divide_df(my_df)

    if len(df_1) > limit:
        df_1 = convert(df_1)
    if len(df_2) > limit:
        df_2 = convert(df_2)
    list_of_df.append(df_1)
    list_of_df.append(df_2)
    return df_1, df_2

# TODO: same way as sub_level
def add_enron_columns(cat_emails : pd.DataFrame, enron : pd.DataFrame) -> pd.DataFrame:
    """
    Add the enron columns to the dataframe
     """
    cat_emails['ID'] = enron['Message-ID']
    cat_emails['Subject'] = enron['Subject']
    cat_emails['enron_content'] = enron['content']
    cat_emails['From'] = enron['From']
    cat_emails['To'] = enron['To']
    cat_emails['Cc'] = enron['X-cc']
    cat_emails['Bcc'] = enron['X-bcc']
    cat_emails.rename(columns={0: 'Lemmatized_Text'}, inplace=True)
    return cat_emails

def iterate_levels(list_of_df : list) -> list:
    """
    Iterate through the levels of the topic hierarchy
    list: list of dataframes
    return: a list of dataframes
    boolean: if there are less than 6 emails in the leaf. No more need to iterate
    """
    result = []
    
    for one_df in list_of_df:
        df_res_1, df_res_2 = sub_level(one_df) # returns two dataframes
        result.append(df_res_1)
        result.append(df_res_2)
    return result


def first_level():
    """
    Starting from top level, that's a special case: needs to load files
    """
    data_lemmatized, enron = load_files(test=True)
    id2word = create_id2word(data_lemmatized)
    corpus = calc_term_doc_frequency(id2word, data_lemmatized)
    lda_model = calculate_lda_model(corpus, id2word, 2)

    categorised_emails = categorise_emails(lda_model, corpus, data_lemmatized)
    categorised_emails = add_enron_columns(categorised_emails, enron)
    categorised_emails = convert_topic_and_perc(categorised_emails)

    list_of_df= sub_level(categorised_emails)
    
    # Obiviously there should be loop here, but for now it's fine as 
    # we are testing
    result = iterate_levels(list_of_df)
    result = iterate_levels(result)
    result = iterate_levels(result)
    result = iterate_levels(result)
    result = iterate_levels(result)
    print("RESULT : ", result)



    # Combine the list of DataFrames to one dataframe

    categorised_emails = pd.concat(result, axis = 0)
    # write datafreame to csv
    with open('./result/result.csv', 'w') as my_file:
        categorised_emails.to_csv(my_file, index=False)

    #calculate_score(corpus, lda_model, id2word, data_lemmatized)



if __name__ == '__main__':
    first_level()
