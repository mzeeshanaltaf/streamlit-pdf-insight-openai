from util import *


st.title('Query Documents 🔍📄')
st.write(":blue[*Engage with the uploaded documents through a chat interface or "
         "pre-defined prompts. Ask specific questions about the content of the documents and receive "
         "precise answers, facilitating a deeper understanding and efficient information retrieval.*]")

# Display the list of predefined prompts and return the prompt detail which user has selected
predefined_query = get_predefined_prompts()

# Get LLM and Prompt
llm = get_llm()
prompt = get_prompt()

st.divider()

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

container = st.container(border=True)
question = st.chat_input(placeholder='Enter your question related to uploaded document')
if question or predefined_query:
    content = question if question else predefined_query
    st.session_state.messages.append({"role": "user", "content": content})
    st.chat_message("user").write(content)

    with st.spinner('Processing...'):
        st.session_state.response = get_llm_response(llm, prompt, content)
        st.session_state.messages.append({"role": "assistant", "content": st.session_state.response['answer']})
        st.chat_message("assistant").write(st.session_state.response['answer'])
