from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import streamlit as st
import pandas as pd
from pdf2image import convert_from_path
import tempfile
import os
import re
import json
from streamlit_tags import st_tags


# Function for API configuration at sidebar
def sidebar_api_key_configuration():
    st.sidebar.subheader("API Keys")
    openai_api_key = st.sidebar.text_input("Enter your OpenAI API Key 🗝️", type="password",
                                           help='Get OpenAI API Key from: https://platform.openai.com/api-keys')
    groq_api_key = st.sidebar.text_input("Enter your Groq API Key 🗝️", type="password",
                                         help='Get Groq API Key from: https://console.groq.com/keys')
    if openai_api_key == '' and groq_api_key == '':
        st.sidebar.warning('Enter the API Key(s) 🗝️')
        st.session_state.prompt_activation = False
    elif (openai_api_key.startswith('sk-') and (len(openai_api_key) == 56)) and (groq_api_key.startswith('gsk_') and
                                                                                 (len(groq_api_key) == 56)):
        st.sidebar.success('Lets Proceed!', icon='️👉')
        st.session_state.prompt_activation = True
    else:
        st.sidebar.warning('Please enter the correct API Key 🗝️!', icon='⚠️')
        st.session_state.prompt_activation = False
    return openai_api_key, groq_api_key


def sidebar_groq_model_selection():
    st.sidebar.subheader("Model Selection")
    model = st.sidebar.selectbox('Select the Model', ('Llama3-8b-8192', 'Llama3-70b-8192', 'Mixtral-8x7b-32768',
                                                      'Gemma-7b-it'), label_visibility="collapsed")
    return model


# Read PDF data
def read_pdf_data(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


# Split data into chunks
def split_data(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    text_chunks = text_splitter.split_text(text)
    return text_chunks


# Create vectorstore
def create_vectorstore(openai_api_key, pdf_docs):
    raw_text = read_pdf_data(pdf_docs)  # Get PDF text
    text_chunks = split_data(raw_text)  # Get the text chunks
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    # embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


# Get response from llm of user asked question
def get_llm_response(llm, prompt, question):
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(st.session_state.vector_store.as_retriever(), document_chain)
    response = retrieval_chain.invoke({'input': question})
    return response


def generate_summary(llm, prompt, pdf_docs):
    question = ('Provide the following information in JSON format: '
                '1) Document Category '
                '2) Top 5 keywords '
                '3) Summarize this document'
                'JSON format should following key value pairs: category: <Document Category>, '
                'keywords: <Top 5 Keywords>'
                'summary: <Document Summary> and starts and ends with 3 backticks. Do not use json keyword'
                'in the response')

    # Initial empty DataFrame
    df = pd.DataFrame(columns=['Filename', 'Filesize', 'Category', 'Keywords', 'Summary'])

    for pdf in pdf_docs:
        st.session_state.vector_store = create_vectorstore(st.session_state.openai_api_key, [pdf])
        response_json = get_llm_response(llm, prompt, question)
        response = response_json['answer']
        print(response)
        # Remove backticks using regular expressions
        json_string = response.strip().strip("`")

        # Replace keys without quotes with quoted keys
        json_string = re.sub(r'(\w+):', r'"\1":', json_string)
        print(json_string)
        json_data = json.loads(json_string)
        data = {
            'Filename': pdf.name,
            'Filesize': pdf.size,
            'Category': json_data['category'],
            'Keywords': json_data['keywords'],
            'Summary': json_data['summary'],
        }
        data_df = pd.DataFrame([data])
        df = pd.concat([df, data_df])
    return df


# Function to extract thumbnail image from PDF
def extract_pdf_thumbnail(pdf_file_path):
    images = convert_from_path(pdf_file_path, first_page=0, last_page=1)
    first_page_image = images[0]

    # Convert the image to thumbnail
    first_page_image.thumbnail((200, 200))
    return first_page_image


def pdf_thumbnails(uploaded_files):
    thumbnails = []
    # Loop through each uploaded file and display in 3x3 grid
    for uploaded_file in uploaded_files:
        # Create a temporary file to save the uploaded PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf.write(uploaded_file.read())
            temp_pdf_path = temp_pdf.name

        # Extract thumbnail from PDF file
        thumbnail = extract_pdf_thumbnail(temp_pdf_path)

        # Append the thumbnails in list
        thumbnails.append(thumbnail)

        # Remove the temporary files
        os.remove(temp_pdf_path)

    return thumbnails


def display_thumbnails(thumbnails, df):

    # Calculate the number of rows needed
    n_files = len(thumbnails)
    n_cols = 3
    n_rows = (n_files + n_cols - 1) // n_cols  # Ceiling division

    # Create a list to hold columns for layout
    cols = [st.columns(n_cols) for _ in range(n_rows)]

    for i, thumbnail in enumerate(thumbnails):
        col = cols[i // n_cols][i % n_cols]

        with col:
            # Display the image using Streamlit
            st.image(thumbnail, use_column_width=False)

            with st.popover("Details"):
                st.markdown(f"File name: {df.iloc[i]['Filename']}")
                size = round(df.iloc[i]['Filesize'] / (1024 * 1024), 2)
                st.markdown(f"File size: {size} MB")
                st.markdown(f"Category:  {df.iloc[i]['Category']}")
                st.markdown(f"Keywords:  ")
                st_tags(label='', text='', value=df.iloc[i]['Keywords'], maxtags=10, key=i)

                st.markdown(f"Summary:  {df.iloc[i]['Summary']}")



