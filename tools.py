import re

from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.settings import Settings
from llama_index.core.tools import FunctionTool
from googlesearch import search as google_search
from config import Config

def enforce_prefixes(llm_output):
    date_pattern = re.compile(r"^\(\d{4}(?:-\d{2}-\d{2})?\)")
    lines = [line.strip() for line in llm_output.split('\n') if line.strip()]
    processed = []
    for line in lines:
        if date_pattern.match(line):
            processed.append(line)
        else:
            processed.append(f"(Date not specified) {line}")
    return "\n".join(processed)

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

            # Fixed: Use a list to collect cleaned results
            cleaned_results = []
            for result in results:
                description = getattr(result, 'snippet', None) or getattr(result, 'description', None)
                if description:
                    cleaned = self._clean_search_result(description)
                    if cleaned:
                        cleaned_results.append(cleaned)

            if not cleaned_results:
                return "No relevant summaries found."

            # Join with triple newlines for better separation and readability
            cleaned_paragraphs = "\n\n\n".join(cleaned_results)

            # Enhanced prompt to preserve policy dates and timing
            prompt = (
                "Please organize the following search results into clear, factual bullet points. "
                "IMPORTANT: Preserve all policy update dates, effective dates, and timing information. "
                "Include when policies were announced, implemented, or became effective. "
                "Format each point clearly with dates in brackets when available. "
                "Avoid ellipses, vague terms, or broken phrases:\n\n"
                + cleaned_paragraphs
            )

            response = self.llm.complete(prompt)
            return str(response)  # Ensure it's a string

        except Exception as e:
            return f"Search failed: {str(e)}"

    def _clean_search_result(self, text: str) -> str:
        """Extract policy dates and filter relevant sentences with proper line breaks"""
        import re
        import os

        if not text:
            return ""

        # Store original text for date extraction
        original_text = text
        text = text.replace("...", "")  # Remove ellipses

        # Extract policy update dates and timing information
        policy_dates = []
        
        # Comprehensive date patterns for policy updates
        policy_date_patterns = [
            r'(?:updated|effective|announced|implemented|launched|revised|introduced|started|beginning)\s+(?:on\s+)?(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            r'(?:updated|effective|announced|implemented|launched|revised|introduced|started|beginning)\s+(?:on\s+)?(\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{2,4})',
            r'(?:updated|effective|announced|implemented|launched|revised|introduced|started|beginning)\s+(?:on\s+)?(\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{2,4})',
            r'(?:from|since|as of)\s+(\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{2,4})',
            r'(?:from|since|as of)\s+(\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{2,4})',
            r'(\d{4})\s+(?:budget|policy|update|changes)',
            r'(?:in|for)\s+(\d{4})',
        ]

        # Extract dates related to policies
        for pattern in policy_date_patterns:
            matches = re.findall(pattern, original_text, flags=re.IGNORECASE)
            policy_dates.extend(matches)

        # Remove non-policy related metadata patterns only
        metadata_patterns = [
            r'^\d+\s+(hours?|days?|weeks?|months?)\s+ago\s*[·•-]\s*',
            r'^\w+\s*[·•-]\s*(?!\s*(?:updated|effective|announced|policy))',  # Don't remove if followed by policy terms
            r'\s*[·•-]\s*\d+\s+(hours?|days?|weeks?|months?)\s+ago.*\]',
        ]

        # Only remove time patterns, not date patterns that might be policy-related
        time_patterns = [
            r'\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm)?\b',
            r'\b\d{1,2}\s*(?:AM|PM|am|pm)\b',
        ]

        for pattern in time_patterns + metadata_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)

        # Enhanced keywords to capture policy timing and updates
        keywords = [
            'policy', 'budget', 'government', 'change', 'support', 'economy', 
            'families', 'workers', 'jobs', 'benefit', 'insurance', 'health', 
            'cost', 'coverage', 'updated', 'effective', 'announced', 'implemented',
            'launched', 'revised', 'introduced', 'started', 'beginning', 'from',
            'since', 'as of', 'new', 'latest', 'recent', 'current'
        ]

        sentences = re.split(r'(?<=[.?!])\s+', text)
        important = []

        for sentence in sentences:
            sentence = sentence.strip()
            if any(k in sentence.lower() for k in keywords):
                if len(sentence) < 15:  # Slightly lower threshold for policy dates
                    continue
                if 'from .' in sentence:
                    continue
                # Don't skip sentences with ellipses if they contain policy information
                if '...' in sentence and not any(policy_word in sentence.lower() for policy_word in ['policy', 'updated', 'effective', 'announced']):
                    continue
                
                # Add date context if available
                sentence_with_context = sentence
                if policy_dates:
                    # Check if this sentence already contains a date
                    has_date = any(re.search(r'\b\d{4}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)', sentence, re.IGNORECASE) for date in policy_dates)
                    if not has_date and any(policy_word in sentence.lower() for policy_word in ['policy', 'benefit', 'coverage', 'support']):
                        # Add the most recent policy date as context
                        latest_date = policy_dates[-1] if policy_dates else None
                        if latest_date:
                            sentence_with_context = f"[{latest_date}] {sentence}"
                
                important.append(sentence_with_context)

        # Add policy dates summary at the beginning if found
        result_parts = []
        if policy_dates:
            unique_dates = list(set(policy_dates))
            if unique_dates:
                result_parts.append(f"Policy Update Dates: {', '.join(unique_dates)}")
        
        if important:
            result_parts.extend(important)

        # Use double line breaks for better readability
        return (os.linesep + os.linesep).join(result_parts)

    def get_search_tool(self):
        """Return the search tool for external use"""
        return self.search_tool