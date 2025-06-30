import streamlit as st
from tools import Tools 
from styles import get_custom_css
from utils import render_header, render_footer, render_results

# Initialize AI tools
@st.cache_resource
def initialize_ai_tools():
    return Tools()

def main():
    # Page configuration
    st.set_page_config(
        page_title="PRUClarity", 
        page_icon="🤖",
        layout="wide"
    )
    
    # Apply custom CSS
    st.markdown(get_custom_css(), unsafe_allow_html=True)
    
    # Initialize AI tools
    ai_tools = initialize_ai_tools()
    search_tool = ai_tools.get_search_tool()
    
    # Render header
    render_header()
    
    # Input Section
    st.markdown('<label class="input-label">Ask about financial policies or market news:</label>', unsafe_allow_html=True)
    query = st.text_input("", placeholder="What are the new policy changes?", label_visibility="collapsed")
    
    # Search and display results
    if query:
        with st.spinner("🔍 Searching for the latest information..."):
            search_result = search_tool.call(query).content
        
        render_results(search_result)
    
    # Render footer
    render_footer()

if __name__ == "__main__":
    main()