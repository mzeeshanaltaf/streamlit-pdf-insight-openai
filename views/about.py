import streamlit as st

st.subheader('About')
with st.expander('Application'):
    st.markdown(''' Get to know the insights of your documents.''')
with st.expander('Technologies Used'):
    st.markdown(''' 
    * OpenAI LLM -- For Q&A with documents
    * OpenAI Embeddings -- For creating Embeddings for Vector DB
    * Pinecone -- For storing embeddings in Vector DB
    * Streamlit -- For application Front End
    ''')
with st.expander('Contact'):
    st.markdown(''' Any Queries: Contact [Zeeshan Altaf](mailto:zeeshan.altaf@gmail.com)''')
with st.expander('Source Code'):
    st.markdown(''' Source code: [GitHub](https://github.com/mzeeshanaltaf/)''')
