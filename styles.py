def get_custom_css():
    """Return custom CSS for Prudential-inspired theme"""
    return """
    <style>
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        /* Main App Background */
        .stApp {
            background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        }
        
        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* Custom Header Styling */
        .main-header {
            background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%);
            padding: 2rem 1rem;
            border-radius: 15px;
            margin-bottom: 2rem;
            box-shadow: 0 10px 25px rgba(220, 38, 38, 0.15);
            border: 1px solid rgba(220, 38, 38, 0.1);
        }
        
        .main-header h1 {
            color: white !important;
            font-family: 'Inter', sans-serif !important;
            font-weight: 700 !important;
            font-size: 2.5rem !important;
            margin: 0 !important;
            text-align: center;
            text-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .main-header .subtitle {
            color: rgba(255, 255, 255, 0.9) !important;
            font-family: 'Inter', sans-serif !important;
            font-weight: 400 !important;
            font-size: 1.1rem !important;
            text-align: center;
            margin-top: 0.5rem;
        }
        
        /* Input Field Styling */
        .stTextInput > div > div > input {
            background-color: white;
            border: 2px solid #e5e7eb;
            border-radius: 10px;
            padding: 0.75rem 1rem;
            font-family: 'Inter', sans-serif;
            font-size: 1rem;
            color: #000000;
            transition: all 0.3s ease;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        
        .stTextInput > div > div > input:focus {
            border-color: #dc2626;
            box-shadow: 0 0 0 3px rgba(220, 38, 38, 0.1);
            outline: none;
        }
        
        /* Search Button Styling */
        .stButton > button {
            background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%);
            color: white;
            border: none;
            border-radius: 10px;
            padding: 0.75rem 2rem;
            font-family: 'Inter', sans-serif;
            font-weight: 600;
            font-size: 1rem;
            transition: all 0.3s ease;
            box-shadow: 0 4px 12px rgba(220, 38, 38, 0.2);
            width: 100%;
        }
        
        .stButton > button:hover {
            background: linear-gradient(135deg, #b91c1c 0%, #991b1b 100%);
            transform: translateY(-2px);
            box-shadow: 0 6px 16px rgba(220, 38, 38, 0.3);
        }
        
        /* Results Card Styling */
        .results-card {
            background: white;
            border-radius: 15px;
            padding: 2rem;
            margin: 1.5rem 0;
            box-shadow: 0 8px 25px rgba(0,0,0,0.08);
            border: 1px solid rgba(220, 38, 38, 0.1);
            border-left: 5px solid #dc2626;
        }
        
        /* Subheader Styling */
        .custom-subheader {
            color: #dc2626;
            font-family: 'Inter', sans-serif;
            font-weight: 600;
            font-size: 1.5rem;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid rgba(220, 38, 38, 0.1);
        }
        
        /* Results Text Styling */
        .results-text {
            font-family: 'Inter', sans-serif;
            font-size: 1rem;
            line-height: 1.6;
            color: #000000;
            background: #f9fafb;
            padding: 1.5rem;
            border-radius: 10px;
            border-left: 4px solid #dc2626;
        }
        
        /* Spinner Styling */
        .stSpinner > div {
            border-top-color: #dc2626 !important;
        }
        
        /* Sidebar Styling */
        .css-1d391kg {
            background: linear-gradient(135deg, #fef2f2 0%, #ffffff 100%);
        }
        
        /* Custom Input Label */
        .input-label {
            color: #000000;
            font-family: 'Inter', sans-serif;
            font-weight: 500;
            font-size: 1.1rem;
            margin-bottom: 0.5rem;
            display: block;
        }
        
        /* Loading Animation */
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        
        .loading-text {
            animation: pulse 1.5s ease-in-out infinite;
            color: #dc2626;
            font-weight: 600;
        }
    </style>
    """
