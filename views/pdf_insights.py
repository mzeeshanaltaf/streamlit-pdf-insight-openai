from util import *

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="PDF Insights", page_icon="✨", layout="centered")

# --- SETUP SESSION STATE VARIABLES ---
if "vector_store_db" not in st.session_state:
    st.session_state.vector_store_db = None
if "prompt_list" not in st.session_state:
    st.session_state.prompt_list = []
if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame(columns=['Filename', 'Filesize', 'Category', 'Keywords', 'Summary'])
if "thumbnails" not in st.session_state:
    st.session_state.thumbnails = []
if "response" not in st.session_state:
    st.session_state.response = None
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = None
if "prompt" not in st.session_state:
    st.session_state.prompt = False
if "pdf_docs" not in st.session_state:
    st.session_state.pdf_docs = None
if "summary" not in st.session_state:
    st.session_state.summary = None

# --- MAIN PAGE CONFIGURATION ---
st.title("PDF Insights 📄✨")
st.write("*Interrogate Documents :books:, Ignite Insights: AI at Your Service*")

# Get LLM and Prompt
llm = get_llm()
prompt = get_prompt()

# ----- SETUP PDF GENIE MENU ------

st.subheader("Upload PDF(s)")
pdf_docs = st.file_uploader("Upload your PDFs", type='pdf', accept_multiple_files=True,
                            label_visibility='collapsed')

summary = st.button("Process", type="primary", key="process", disabled=not pdf_docs)
if summary:
    with st.spinner("Processing ..."):

        # Generate the list of PDF(s) thumbnails
        pdf_thumbnails(pdf_docs)

        # Analyze PDF(s) using LLM and store the summary in data frame
        analyze_documents(llm, prompt, pdf_docs)

        # Display the success message and a notification toast
        st.success('Processing Completed Successfully')
        st.toast('Processing Completed Successfully', icon='🎉')

