from pypdf import PdfReader
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
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
import json
from streamlit_tags import st_tags


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
def create_vectorstore(pdf_docs):
    raw_text = read_pdf_data(pdf_docs)  # Get PDF text
    text_chunks = split_data(raw_text)  # Get the text chunks
    embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"])
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


# Get response from llm of user asked question
def get_llm_response(llm, prompt, question):
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(st.session_state.vector_store.as_retriever(), document_chain)
    response = retrieval_chain.invoke({'input': question})
    return response


def analyze_documents(llm, prompt, pdf_docs):
    question = ('Provide the following information in JSON format: '
                '1) Document Category '
                '2) Top 5 keywords '
                '3) Summarize this document'
                'JSON should have following format: '
                '```'
                '{'
                '"category": "<Document Category>", '
                '"keywords": "[<Top 5 Keywords in double quotes>]"'
                '"summary": "<Document Summary>".'
                '}'
                '```'
                'Do not use json keyword in the response')

    for pdf in pdf_docs:
        st.session_state.vector_store = create_vectorstore([pdf])
        response_json = get_llm_response(llm, prompt, question)
        response = response_json['answer']
        # print(response)

        # Remove backticks from the response
        json_string = response.strip().strip("`")

        # print(json_string)
        json_data = json.loads(json_string)
        data = {
            'Filename': pdf.name,
            'Filesize': pdf.size,
            'Category': json_data['category'],
            'Keywords': json_data['keywords'],
            'Summary': json_data['summary'],
        }
        data_df = pd.DataFrame([data])
        st.session_state.df = pd.concat([st.session_state.df, data_df])


# Function to extract thumbnail image from PDF
def extract_pdf_thumbnail(pdf_file_path):
    images = convert_from_path(pdf_file_path, first_page=0, last_page=1)
    first_page_image = images[0]

    # Convert the image to thumbnail
    first_page_image.thumbnail((200, 200))
    return first_page_image


def pdf_thumbnails(uploaded_files):
    # Loop through each uploaded file and display in 3x3 grid
    for uploaded_file in uploaded_files:
        # Create a temporary file to save the uploaded PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf.write(uploaded_file.read())
            temp_pdf_path = temp_pdf.name

        # Extract thumbnail from PDF file
        thumbnail = extract_pdf_thumbnail(temp_pdf_path)

        # Append the thumbnails in list
        st.session_state.thumbnails.append(thumbnail)

        # Remove the temporary files
        os.remove(temp_pdf_path)


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

            with st.popover("Details", use_container_width=False):
                st.markdown(f"File name: {df.iloc[i]['Filename']}")
                size = round(df.iloc[i]['Filesize'] / (1024 * 1024), 2)
                st.markdown(f"File size: {size} MB")
                st.markdown(f"Category:  {df.iloc[i]['Category']}")
                st.markdown(f"Keywords:  ")
                st_tags(label='', text='', value=df.iloc[i]['Keywords'], maxtags=10, key=i)

                st.markdown(f"Summary:  {df.iloc[i]['Summary']}")


def get_predefined_prompts():
    st.subheader("Pre-defined Prompts:")
    prompt_details = None
    # Calculate the number of rows needed
    n_files = len(st.session_state.prompt_list)
    n_cols = 5
    n_rows = (n_files + n_cols - 1) // n_cols  # Ceiling division

    # Create a list to hold columns for layout
    cols = [st.columns(n_cols) for _ in range(n_rows)]

    for i in range(len(st.session_state.prompt_list)):
        col = cols[i // n_cols][i % n_cols]
        with col:
            button_press = st.button(st.session_state.prompt_list[i]['title'], key=i, use_container_width=True)
            if button_press:
                prompt_details = st.session_state.prompt_list[i]['details']

    return prompt_details


def get_llm():
    return ChatOpenAI(openai_api_key=st.secrets["OPENAI_API_KEY"], model="gpt-3.5-turbo")


def get_prompt():
    prompt = ChatPromptTemplate.from_template(
        """
        Answer the question based on the provided context only. If question is not within the context, do not try to answer
        and respond that the asked question is out of context or something similar.
        Please provide the most accurate response based on the question.
        <context>
        {context}
        Questions: {input}
        """
    )
    return prompt
