import streamlit as st

def render_header():
    """Render the main header"""
    st.markdown("""
    <div class="main-header">
        <h1>PRUClarity</h1>
        <div class="subtitle">Your AI-Powered Financial Assistant</div>
    </div>
    """, unsafe_allow_html=True)

def render_footer():
    """Render the footer"""
    st.markdown("""
    <div style="margin-top: 3rem; padding: 2rem; text-align: center; color: #000000; font-family: 'Inter', sans-serif;">
        <p>Powered by AI • Built for Financial Insights</p>
    </div>
    """, unsafe_allow_html=True)

def render_results(search_result):
    """Render search results in a styled card"""
    st.markdown('<div class="results-card">', unsafe_allow_html=True)
    st.markdown('<h3 class="custom-subheader">📊 Search Results</h3>', unsafe_allow_html=True)
    st.markdown(f'<div class="results-text">{search_result}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)