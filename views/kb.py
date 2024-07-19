import streamlit as st
from util import *

st.title('Knowledge Base')
st.write(":blue[*This page serves as a comprehensive archive for easy reference and retrieval of document "
         "information. Browse, search, and access summaries, important keywords, and categories of previously uploaded "
         "documents.*]")

# Display PDF(s) thumbnails along with summary
display_thumbnails(st.session_state.thumbnails, st.session_state.df)
