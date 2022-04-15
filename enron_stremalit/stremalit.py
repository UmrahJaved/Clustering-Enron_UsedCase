import streamlit as st
import pandas as pd
import numpy as np

with st.sidebar:
  
  page = st.radio(
      "",
      ('Introduction', 'LDA Model'))


if page == 'Introduction':
  with st.container():
    st.image("/home/becode2/enron_DF/enron_stremalit/Analyzing-Enron-Blog-Header-v1.png")
    st.title('Enron Topic Analysis')
    with st.container():
        st.write("""Enron Corporation was an American energy, commodities, and services company based in Houston, Texas. Before its bankruptcy on December 2, 2001, Enron employed approximately 20,000 staff and was
                  one of the world’s major electricity, natural gas, communications, and
                  pulp and paper companies, with claimed revenues of nearly $111 billion during 2000. At the end of 2001, it was revealed that its reported
                  financial condition was sustained substantially by an institutionalized,
                  systematic, and creatively planned accounting fraud, known since as
                  the Enron scandal. Enron has since become a well-known example of
                  wilful corporate fraud and corruption. This report aims at answering
                  whether top level Enron employees had incriminating evidence in their
                  office emails or uncover any unusual patterns in the months leading
                  up to the scandal through an exploratory data analysis.""")

elif page == 'LDA Model':
  with st.container():
      st.header("Topic Modelling with LDA")
      st.write("""Latent Dirichlet Allocation was performed on the dataset with the number
  of topics, k = X. After selectinc a topic, the following is a list of the to terms for each of the topics 
  and finally a dataframe containing the top emails related to these terms""")

  with st.container():
    df = pd.read_csv('/home/becode2/enron_DF/enron_stremalit/result.csv')
    df2 = df.drop(columns=['Dominant_Topic'])
    first_level = st.container()
    second_level = st.container()
    third_level = st.container()
    df3 = df['enron_content'].loc[df["Topic_hier"].str.startswith('[0') == True].head(20)
    df4 = df['enron_content'].loc[df["Topic_hier"].str.startswith('[1') == True].head(20)
    topics = ["Enron ", "Market", "Comunication", "Law"]
    subtopics = ['Phillip', 'Accounting', 'Comunication','Gas', 'Closing', 'Business','Law','Documents']
    x1= "Phillip"
    x2= "Accounting"
    x3= ""
    x4= " "
    
    with first_level:
      col1, col2, col3, col4 = st.columns(4)
      st.header("")

      with col1:
        if st.button(topics[0]):
          x1 = subtopics[0]
          x2 = subtopics[1]
      with col2:
        if st.button(topics[1]):
          x1 = subtopics[2]
          x2 = subtopics[3]
      with col3:
        if st.button(topics[2]):
          x1 = subtopics[4]
          x2 = subtopics[5]
      with col4:
        if st.button(topics[3]):
          x1 = subtopics[6]
          x2 = subtopics[7]
  with second_level:
    col1_1, col2_1, col3_1, col_4_1  = st.columns(4)
    with col2_1:
      if st.button(x1):
        x3 = 'Phillip'
        x4 = 'Closing'
    with col3_1:
      if st.button(x2):
        x3 = 'Accounting'
        x4 = 'Business'
  with third_level:
    with st.expander('EMAILS'):
      if x3 == 'Phillip':
        st.dataframe(df3) 
      elif x3 == 'Accounting':
        st.dataframe(df4)
      elif x4 == 'Closing':
        st.dataframe(df4)
      elif x4 == 'Business':
        st.dataframe(df4)
