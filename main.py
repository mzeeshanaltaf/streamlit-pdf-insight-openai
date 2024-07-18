from util import *
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="PDF Insights", page_icon="✨", layout="centered")


# --- SETUP SESSION STATE VARIABLES ---
if "vector_store" not in st.session_state:
    st.session_state.vector_store = False
if "response" not in st.session_state:
    st.session_state.response = None
if "prompt_activation" not in st.session_state:
    st.session_state.prompt_activation = False
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = None
if "prompt" not in st.session_state:
    st.session_state.prompt = False
if "pdf_docs" not in st.session_state:
    st.session_state.pdf_docs = None
if "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = None
if "groq_api_key" not in st.session_state:
    st.session_state.groq_api_key = None
if "summary" not in st.session_state:
    st.session_state.summary = None
if "df" not in st.session_state:
    st.session_state.df = None
if "thumbnails" not in st.session_state:
    st.session_state.thumbnails = None

load_dotenv()
st.session_state.openai_api_key = os.environ.get('OPENAI_API_KEY')

# --- MAIN PAGE CONFIGURATION ---
st.title("PDF Insights 📄✨")
st.write("*Interrogate Documents :books:, Ignite Insights: AI at Your Service*")

llm = ChatOpenAI(openai_api_key=st.session_state.openai_api_key, model="gpt-3.5-turbo")

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
# ----- SETUP PDF GENIE MENU ------

st.subheader("Upload PDF(s)")
pdf_docs = st.file_uploader("Upload your PDFs", type='pdf', accept_multiple_files=True,
                            label_visibility='collapsed')

summary = st.button("Get Insight", type="primary", key="process", disabled=not pdf_docs)
if summary or st.session_state.summary:
    with st.spinner("Processing ..."):
        st.session_state.summary = True

        if summary:

            # Get the list of PDF(s) thumbnails
            st.session_state.thumbnails = pdf_thumbnails(pdf_docs)

            # Analyze PDF(s) using LLM and store the summary in data frame
            st.session_state.df = generate_summary(llm, prompt, pdf_docs)

        st.subheader('Knowledge Base:')
        # Display PDF(s) thumbnails along with summary
        display_thumbnails(st.session_state.thumbnails, st.session_state.df)

