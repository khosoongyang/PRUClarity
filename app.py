import streamlit as st
from tools import Tools 
from styles import get_custom_css
from utils import render_footer, render_results

# Initialize AI tools
@st.cache_resource
def initialize_ai_tools():
    return Tools()

def nav_bar():
    nav_options = ["Home", "About Us"]
    if "nav_page" not in st.session_state:
        st.session_state["nav_page"] = nav_options[0]

    st.markdown(
        """
        <style>
            .custom-nav-outer {
                width: 100vw;
                position: fixed;
                top: 0;
                left: 0;
                right: 0;
                z-index: 1000;
                margin: 0;
                padding: 0;
                background: #dc2626;
                font-family: 'Poppins', sans-serif !important;
            }
            .custom-nav {
                display: flex;
                justify-content: flex-start;
                align-items: center;
                gap: 2rem;
                background: transparent;
                border-radius: 0;
                padding: 0.5rem 2vw 0.5rem 2vw;
                width: 100vw;
                box-shadow: none;
                margin: 0;
                font-family: 'Poppins', sans-serif !important;
            }
            .custom-nav-btn {
                font-family: 'Poppins', sans-serif !important;
                font-weight: 600;
                font-size: 1.06rem;
                color: #fff !important;
                background: none;
                border: none;
                padding: 0.4rem 1.4rem;
                border-radius: 6px;
                cursor: pointer;
                transition: background 0.18s, color 0.18s;
                margin-top: 0.1rem;
                margin-bottom: 0.1rem;
            }
            .custom-nav-btn.selected, .custom-nav-btn:hover {
                background: #fff !important;
                color: #dc2626 !important;
            }
            .stApp {
                padding-top: 2.7rem !important; /* Reduce top padding further */
                font-family: 'Poppins', sans-serif !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    nav_html = '<div class="custom-nav-outer"><div class="custom-nav">'
    for nav in nav_options:
        selected = "selected" if st.session_state["nav_page"] == nav else ""
        nav_html += (
            f'<form action="" method="post" style="display:inline;">'
            f'<button name="nav_{nav}" type="submit" class="custom-nav-btn {selected}">{nav}</button>'
            f'</form>'
        )
    nav_html += '</div></div>'
    st.markdown(nav_html, unsafe_allow_html=True)

    # Button state management
    for nav in nav_options:
        nav_key = f"nav_{nav}"
        if st.session_state.get(nav_key, False):
            st.session_state["nav_page"] = nav
            st.session_state[nav_key] = False

def render_header_with_logo():
    st.markdown(
        """
        <style>
        .main-header {
            background: linear-gradient(135deg, #dc2626 0%, #dc2626 100%) !important;
            padding: 1.6rem 1rem 1.1rem 1rem; /* Even less padding */
            border-radius: 15px;
            margin-bottom: 0.7rem; /* Less gap below */
            margin-top: 0.1rem;   /* Less gap above */
            box-shadow: 0 12px 24px rgba(0, 0, 0, 0.1);
            text-align: center;
            font-family: 'Poppins', sans-serif !important;
            position: relative;
        }
        .main-header h1 {
            color: white;
            font-weight: 700;
            font-size: 2.5rem;
            text-shadow: 0 2px 4px rgba(0, 0, 0, 0.15);
            margin: 0;
            font-family: 'Poppins', sans-serif !important;
        }
        .main-header .subtitle {
            color: rgba(255, 255, 255, 0.9);
            font-weight: 400;
            font-size: 1.1rem;
            margin-top: 0.5rem;
            margin-bottom: 0.4rem;
            font-family: 'Poppins', sans-serif !important;
        }
        .prudential-logo {
            margin-top: 0.5rem;
            margin-bottom: 0rem;
            max-width: 110px;
            width: 28vw;
        }
        </style>
        """, unsafe_allow_html=True
    )
    st.markdown(
        """
        <div class="main-header">
            <h1>PRUClarity</h1>
            <div class="subtitle">Your AI-Powered Financial Assistant</div>
            <img class="prudential-logo" src="https://upload.wikimedia.org/wikipedia/commons/4/4b/Prudential_plc_logo.svg" alt="Prudential Logo"/>
        </div>
        """,
        unsafe_allow_html=True
    )

def main():
    st.set_page_config(
        page_title="PRUClarity", 
        page_icon="🤖",
        layout="wide"
    )
    st.markdown(get_custom_css(), unsafe_allow_html=True)
    nav_bar()
    # Spacer for fixed navigation bar, reduce it further!
    st.markdown("<div style='height: 1.1rem;'></div>", unsafe_allow_html=True)

    ai_tools = initialize_ai_tools()
    search_tool = ai_tools.get_search_tool()

    # Render header with logo (custom)
    render_header_with_logo()

    if st.session_state["nav_page"] == "Home":
        st.markdown('<label class="input-label">Ask about financial policies or market news:</label>', unsafe_allow_html=True)
        query = st.text_input("", placeholder="What are the new policy changes?", label_visibility="collapsed")
        if query:
            with st.spinner("🔍 Searching for the latest information..."):
                search_result = search_tool.call(query).content
            render_results(search_result)
    elif st.session_state["nav_page"] == "About Us":
        st.markdown(
            """
            ## About PRUClarity
            PRUClarity is your AI-powered assistant for up-to-date financial policy and market news.  
            - **Mission:** Make complex financial information accessible and clear.
            - **Features:** Instant search, policy explanation, and market news summaries.
            - **Contact:** [info@pruclarity.com](mailto:info@pruclarity.com)
            """
        )
    render_footer()

if __name__ == "__main__":
    main()

