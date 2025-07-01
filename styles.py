def get_custom_css():
    """Return custom CSS for a Prudential-inspired Streamlit theme using Poppins, with a solid red navigation bar."""
    return """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
        html, body, [class*="css"], .stApp {
            font-family: 'Poppins', sans-serif !important;
        }
        .stApp {
            background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
            padding-top: 3.5rem !important;
        }

        #MainMenu, footer, header {
            visibility: hidden;
        }

        /* Header Section */
        .main-header {
            background: linear-gradient(135deg, #dc2626 0%, #dc2626 100%) !important;
            padding: 2rem 1rem;
            border-radius: 15px;
            margin-bottom: 2rem;
            box-shadow: 0 12px 24px rgba(0, 0, 0, 0.1);
            text-align: center;
            font-family: 'Poppins', sans-serif !important;
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
            font-family: 'Poppins', sans-serif !important;
        }
        
        input[type="text"], textarea {
            background-color: #ffffff !important;
            color: #111827 !important;
            border: 2px solid #e5e7eb;
            border-radius: 10px;
            padding: 0.75rem 1rem;
            font-size: 1rem;
            transition: all 0.3s ease;
            box-shadow: 0 2px 6px rgba(0,0,0,0.05);
            font-family: 'Poppins', sans-serif !important;
            caret-color: #dc2626; /* Ensures caret is visible and red */
        }

        input[type="text"]:focus, textarea:focus {
            border-color: #dc2626;
            box-shadow: 0 0 0 4px rgba(220, 38, 38, 0.15);
            outline: none;
        }

        input::placeholder,
        textarea::placeholder {
            color: #9ca3af !important;
            opacity: 1;
            font-family: 'Poppins', sans-serif !important;
        }

        .stButton > button {
            background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%);
            color: white;
            border: none;
            border-radius: 12px;
            padding: 0.75rem 2rem;
            font-weight: 600;
            font-size: 1rem;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 6px 16px rgba(220, 38, 38, 0.25);
            width: 100%;
            font-family: 'Poppins', sans-serif !important;
        }
        .stButton > button:hover {
            background: linear-gradient(135deg, #b91c1c 0%, #991b1b 100%);
            transform: translateY(-2px);
            box-shadow: 0 8px 24px rgba(220, 38, 38, 0.3);
        }
        .results-card {
            background: white;
            border-radius: 15px;
            padding: 2rem;
            margin: 1.5rem 0;
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.08);
            border-left: 5px solid #dc2626;
            font-family: 'Poppins', sans-serif !important;
        }
        .custom-subheader {
            color: #dc2626;
            font-weight: 600;
            font-size: 1.5rem;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid rgba(220, 38, 38, 0.2);
            font-family: 'Poppins', sans-serif !important;
        }
        .results-text {
            background: #f9fafb;
            padding: 1.25rem;
            border-radius: 10px;
            border-left: 4px solid #dc2626;
            font-size: 1rem;
            line-height: 1.6;
            color: #111827;
            font-family: 'Poppins', sans-serif !important;
        }
        .stSpinner > div {
            border-top-color: #dc2626 !important;
        }
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        .loading-text {
            animation: pulse 1.5s ease-in-out infinite;
            color: #dc2626;
            font-weight: 600;
            font-family: 'Poppins', sans-serif !important;
        }
        .input-label {
            color: black !important;
            font-size: 1.05rem;
            font-weight: 500;
            margin-bottom: 0.5rem;
            display: block;
            font-family: 'Poppins', sans-serif !important;
        }
    </style>
    """