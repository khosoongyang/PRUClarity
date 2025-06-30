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
            system_prompt=self.config.system_prompt,
            temperature=0.2,
            max_tokens=512,
        )
        self.embed_model = FastEmbedEmbedding()
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
        Returns clean, structured summaries using LLM refinement.
        """
        try:
            results = google_search(
                query,
                num_results=self.config.MAX_SEARCH_RESULTS,
                lang=self.config.SEARCH_LANGUAGE,
                advanced=True
            )

            cleaned_paragraphs = ""
            for result in results:
                description = getattr(result, 'snippet', None) or getattr(result, 'description', None)
                if description:
                    cleaned = self._clean_search_result(description)
                    if cleaned:
                        cleaned_paragraphs += cleaned + "\n"

            if not cleaned_paragraphs.strip():
                return "No relevant summaries found."

            # Use the LLM to refine and summarize the cleaned points
            prompt = (
                "Please summarize the following search result snippets into clear, factual bullet points. "
                "Avoid ellipses, vague terms, or broken phrases. Keep each point concise, informative, and properly worded:\n\n"
                + cleaned_paragraphs
            )

            response = self.llm.complete(prompt)
            return response

        except Exception as e:
            return f"Search failed: {str(e)}"

    def _clean_search_result(self, text: str) -> str:
        """Remove date/time, source patterns, and filter irrelevant or incomplete sentences"""
        import re

        if not text:
            return ""

        text = text.replace("...", "")  # Remove ellipses

        date_patterns = [
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
            r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b',
            r'\b\d{1,2}\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4}\b',
            r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{2,4}\b',
            r'\b\d{1,2}\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{2,4}\b',
            r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{2,4}\b',
            r'\b\d{1,2}\s+(hours?|days?|weeks?|months?)\s+ago\b',
            r'\byesterday\b',
            r'\btoday\b',
            r'\btomorrow\b',
            r'\blast\s+(week|month|year)\b',
            r'\bnext\s+(week|month|year)\b',
        ]

        time_patterns = [
            r'\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm)?\b',
            r'\b\d{1,2}\s*(?:AM|PM|am|pm)\b',
        ]

        metadata_patterns = [
            r'^\d+\s+(hours?|days?|weeks?|months?)\s+ago\s*[·•-]\s*',
            r'^\w+\s*[·•-]\s*',
            r'\s*[·•-]\s*\d+\s+(hours?|days?|weeks?|months?)\s+ago.*\]',
        ]

        for pattern in date_patterns + time_patterns + metadata_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)

        # Keywords to filter meaningful content
        keywords = ['policy', 'budget', 'government', 'change', 'support', 'economy', 'families', 'workers', 'jobs', 'benefit', 'insurance', 'health', 'cost', 'coverage']

        sentences = re.split(r'(?<=[.?!])\s+', text)
        important = []

        for sentence in sentences:
            sentence = sentence.strip()
            if any(k in sentence.lower() for k in keywords):
                if len(sentence) < 20:  # Too short = likely fragment
                    continue
                if 'from .' in sentence or '(inclusive)' in sentence:
                    continue
                if '...' in sentence:
                    continue
                important.append(sentence)

        return "\n".join(important)

    def get_search_tool(self):
        """Return the search tool for external use"""
        return self.search_tool
