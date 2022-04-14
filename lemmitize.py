"Lemmitizing the data and "


import pandas as pd
import os
from threading import Thread, RLock
import pickle
from nltk.corpus import stopwords
import gensim
from gensim.models.phrases import Phrases, Phraser


lock = RLock()

class Lemmitize(Thread):
    """
    This class is used to lemmitize the data using threads.
    """
    def __init__(self, name, df_to_process):
        Thread.__init__(self)
        self._df_chunk = df_to_process
        self._name = name
    

    def _save_file(self):
        """ 
        saves the _df_chunk file. 
        This is done in a thread-safe way.
        """
        with lock:
            with open(f'./csv/chunks/{self._name}.pkl', 'wb') as my_pickle:
                pickle.dump(self._df_chunk, my_pickle)

    def run(self):
        """
        Thread that lemministize on the chunk of data.
        """
        self._save_file(self._df_chunk)
        print(f"I'm ready {self._name}, the shape of chunk is {self._df_chunk.shape}")


def read_a_file(name: str) -> pd.DataFrame:
    """
    Reads a file
    :param name: the name of the file to read.
    :return: a dataframe.
    """
    with open("./csv/chunks/"+name, 'rb') as my_pickle:
            tmp_df = pickle.load(my_pickle)
            #print(tmp_df)
            return tmp_df

def join_results():
    """
    Joins all the results in a single dataframe.
    """
    joined_df = pd.DataFrame()
    files = os.listdir("./csv/chunks/")
    #print("files: ", files)
    for filename in files:
        if filename.startswith("email"):
            tmp_df = read_a_file(filename)
            joined_df = pd.concat([joined_df, tmp_df])
    with open("./csv/emails_lemmitized.pkl", 'wb') as my_pickle:
        pickle.dump(joined_df, my_pickle)
    #print("joined_df", joined_df)
        
def clean_dataframe(enron: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the dataframe.
    :param df_to_clean: the dataframe to clean.
    :return: the cleaned dataframe.
    """
    enron.drop_duplicates(subset=['content'], inplace=True)
    enron.rename(columns={'content': 'body'}, inplace=True)
    enron = enron[['From', 'To', 'Subject', 'body']]
    stop_words = stopwords.words('english') # will cause error if not installed accordint to the instructions above
    stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'send', 'com']) # if replied the mail contains thes words -> remove them
    data = enron['body'].values.tolist()
    subject = enron['Subject'].values.tolist()

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))



if __name__ == '__main__':
    join = True
    if join:
        join_results()
    else:
        
        number_of_threads = 2
        enron = pd.read_csv('./csv/emails_df_10000.csv') # let's start with a small file
        enron = enron[:5] #Let's take only a small part of the dataframe, easier to test
        enron = clean_dataframe(enron)
        chunk_size = int(len(enron) / number_of_threads)
        for i in range(number_of_threads+1):
            tmp_df = enron[i*chunk_size:(i+1)*chunk_size]
            lemmitize = Lemmitize(f"email_chunk_{i}", tmp_df)
            for i in range(10000000):
                pass
            lemmitize.start()
            lemmitize.join()

    





