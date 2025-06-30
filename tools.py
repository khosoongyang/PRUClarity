from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.settings import Settings
from llama_index.core.tools import FunctionTool
from googlesearch import search as google_search
from config import Config

class Tools:
    def __init__(self):
        self.config = Config()
        self._setup_llm()
        self._setup_tools()
    
    def _setup_llm(self):
        """Initialize LLM and embedding models"""
        self.llm = OpenAI(
            model=self.config.MODEL_NAME,
            api_key=self.config.OPENAI_API_KEY,
        )
        
        self.embed_model = FastEmbedEmbedding()
        
        # Configure global settings
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model
    
    def _setup_tools(self):
        """Initialize AI tools"""
        self.search_tool = FunctionTool.from_defaults(
            fn=self.search,
            name="web_search",
            description="Search the web for current information and news"
        )
    
    def search(self, query: str) -> str:
        """
        Search Google for information related to the query.
        
        Args:
           query: user search query
        Returns:
           context (str): concatenated search results descriptions with dates filtered out
        """
        try:
            results = google_search(
                query, 
                num_results=self.config.MAX_SEARCH_RESULTS, 
                lang=self.config.SEARCH_LANGUAGE, 
                advanced=True
            )
            context = ""
            for result in results:
                description = getattr(result, 'snippet', None) or getattr(result, 'description', None)
                if description:
                    cleaned_description = self._clean_search_result(description)
                    if cleaned_description:
                        context += cleaned_description + "\n\n"
            return context.strip() if context else "No search results found."
        except Exception as e:
            return f"Search failed: {str(e)}"
    
    def _clean_search_result(self, text: str) -> str:
        """
        Clean search result text by removing dates and keeping only main content.
        
        Args:
            text: Raw search result description
        Returns:
            cleaned_text: Text with dates and metadata removed
        """
        import re
        
        if not text:
            return ""
        
        text = text.replace("...", "")  # Remove ellipses (NEW)
        
        # Remove common date patterns
        date_patterns = [
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',  # MM/DD/YYYY, MM-DD-YYYY
            r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b',    # YYYY/MM/DD, YYYY-MM-DD
            r'\b\d{1,2}\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4}\b',  # DD MMM YYYY
            r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{2,4}\b',  # MMM DD, YYYY
            r'\b\d{1,2}\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{2,4}\b',  # DD Month YYYY
            r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{2,4}\b',  # Month DD, YYYY
            r'\b\d{1,2}\s+hours?\s+ago\b',  # X hours ago
            r'\b\d{1,2}\s+days?\s+ago\b',   # X days ago
            r'\b\d{1,2}\s+weeks?\s+ago\b',  # X weeks ago
            r'\b\d{1,2}\s+months?\s+ago\b', # X months ago
            r'\byesterday\b',               # Yesterday
            r'\btoday\b',                   # Today
            r'\btomorrow\b',                # Tomorrow
            r'\blast\s+(week|month|year)\b', # Last week/month/year
            r'\bnext\s+(week|month|year)\b', # Next week/month/year
        ]
        
        # Remove time patterns
        time_patterns = [
            r'\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm)?\b',  # HH:MM, HH:MM:SS, with/without AM/PM
            r'\b\d{1,2}\s*(?:AM|PM|am|pm)\b',  # H AM/PM
        ]
        
        # Remove source/metadata patterns
        metadata_patterns = [
            r'^\d+\s+(hours?|days?|weeks?|months?)\s+ago\s*[·•-]\s*',  # "X days ago •"
            r'^\w+\s*[·•-]\s*',  # "Source •"
            r'\s*[·•-]\s*\d+\s+(hours?|days?|weeks?|months?)\s+ago.*\]'
        ]
        patterns = date_patterns + time_patterns + metadata_patterns
        for pattern in patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)

        keywords = ['policy', 'budget', 'government', 'change', 'support', 'economy', 'families', 'workers', 'jobs']
        
        # Split text into short sentences
        sentences = re.split(r'(?<=[.?!])\s+', text)

        # Keep only short sentences that contain keywords
        important = []
        for sentence in sentences:
            if any(k in sentence.lower() for k in keywords) and len(sentence) <= 160:
                important.append(sentence.strip())

        # Return the most important points
        return " ".join(important)
    

    def get_search_tool(self):
        """Return the search tool for external use"""
        return self.search_tool