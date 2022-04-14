import streamlit as st
import pandas as pd
import numpy as np

with st.sidebar:
  
  page = st.radio(
      "",
      ('Introduction', 'LDA Model','Plots'))


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
    df = pd.read_csv('/home/becode2/enron_DF/enron_stremalit/sample.csv')
    first_level = st.container()
    second_level = st.container()
    topics = ['000','001', '010', '011']
    line_selected = ""
    with first_level:
      col1, col2, col3, col4 = st.columns(4)

      st.header("")

      with col1:
        if st.button(topics[0]):
          line_selected = "/home/becode2/enron_DF/enron_stremalit/Analyzing-Enron-Blog-Header-v1.png"

      with col2:
        if st.button(topics[1]):
          line_selected = "/home/becode2/enron_DF/enron_stremalit/Analyzing-Enron-Blog-Header-v1.png"
      with col3:
        if st.button(topics[2]):
          line_selected =  "/home/becode2/enron_DF/enron_stremalit/Analyzing-Enron-Blog-Header-v1.png"
      with col4:
        if st.button(topics[3]):
          line_selected =  "/home/becode2/enron_DF/enron_stremalit/Analyzing-Enron-Blog-Header-v1.png"

    with second_level:
      img_file = line_selected
      st.image(img_file)

      col1, col2, col3, col4, col5, col6, col7, col8 = st.columns(8)

      subtopics = ['0000', '0001', '0010','0011', '0100', '0101','0110','0111']

      st.header("")
      with col1:
        if st.button(subtopics[0]):
          
      with col2:
        if st.button(topics[1]):
          
      with col3:
        if st.button(topics[2]):

      with col4:
        if st.button(topics[3]):    

      with col5:
        if st.button(topics[4]):

      with col6:
        if st.button(subtopics[5]):
          
      with col7:
        if st.button(topics[6]):
          
      with col8:
        if st.button(topics[7]):

else:
  st.write('done')