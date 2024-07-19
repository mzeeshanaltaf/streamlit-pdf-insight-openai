# Import libraries
import streamlit as st

# --- PAGE SETUP ---
main_page = st.Page(
    "views/pdf_insights.py",
    title="PDF Insights",
    icon=":material/upload_file:",
    default=True,
)

admin_page = st.Page(
    "views/admin.py",
    title="Dashboard",
    icon=":material/admin_panel_settings:",
)

chat_page = st.Page(
    "views/chat.py",
    title="Chat with PDFs",
    icon=":material/chat:",
)

about_page = st.Page(
    "views/about.py",
    title="About",
    icon=":material/info:",
)

kb_page = st.Page(
    "views/kb.py",
    title="Knowledge Base",
    icon=":material/library_books:",
)

pg = st.navigation({
    "Admin": [admin_page],
    "Home": [main_page, kb_page, chat_page],
    "About": [about_page],
                    })

pg.run()
