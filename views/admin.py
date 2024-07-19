from util import *

st.title('Admin Dashboard 📊')
st.write(":blue[*Quick access to statistics, key insights and pre-defined prompts for Admin. "
         "Admin can also create pre-defined prompts.*]")
st.subheader("Statistics:")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(label="Total Documents", value="15")
with col2:
    st.metric(label="Total Users", value="3")
with col3:
    st.metric(label="Number of Prompts", value="50")

st.subheader("Create Pre-defined Prompts:")
prompt_title = st.text_input('Prompt Title:', placeholder="E.g: Machine Learning")
prompt_details = st.text_input('Prompt Details:', placeholder="E.g: Explain Machine Learning to 5 years old")
prompt_generate = st.button('Create Prompt', type='primary', disabled=not prompt_title)

if prompt_generate:
    prompt_dict = {
        "title": prompt_title,
        "details": prompt_details,
    }
    st.session_state.prompt_list.append(prompt_dict)

if st.session_state.prompt_list:
    get_predefined_prompts()


