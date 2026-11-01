import streamlit as st

from portfolio_risk import ui, ui_search

pages = [
    st.Page(ui.main, title="Portfolio risk", url_path="portfolio-risk", default=True),
    st.Page(ui_search.page, title="Add positions", url_path="add-positions"),
]
st.navigation(pages).run()
