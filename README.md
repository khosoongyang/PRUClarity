# PRUClarity


## 🚀 Features

- 🔍 **Live Web Search**: Uses Google search results for up-to-date, real-world context.
- 🧠 **OpenAI GPT-4o-mini**: Delivers smart responses with minimal latency.
- 🧾 **Date-Free Clean Results**: Strips away noise like timestamps and irrelevant metadata.
- 🛠️ **Modular Design**: Tools, styles, and UI logic are well-separated for maintainability.
- 💡 **Fully Customizable**: Easily extend with new tools or models using `llama_index`.

---

## 🗂️ Project Structure

pruclarity/
├── app.py           # Main Streamlit app (frontend)
├── ai_tools.py      # AI tools and LLM configurations  
├── config.py        # Configuration settings
├── styles.py        # CSS styling
├── utils.py         # Utility functions for UI components
└── .env            # Environment variables


## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/pruclarity.git
cd pruclarity
```

### 2. Create and activate a virtual environment
```
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
```

### 3. Install dependencies
```
pip install -r requirements.txt
```

### 4. Create a .env file
```
OPENAI_API_KEY=your_openai_api_key
```

### 🏁 Run the App
```
streamlit run main.py
```

## 📦 Example Usage

### Type a query like:

```
What are the latest CPF policy changes in Singapore?
```
You'll receive a summarized list of web results with cleaned, date-free content to help you quickly digest financial changes.