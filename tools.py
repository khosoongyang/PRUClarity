import re
import os
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
from dataclasses import dataclass

from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.settings import Settings
from llama_index.core.tools import FunctionTool
from googlesearch import search as google_search
from config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Data class for search results"""
    title: str
    url: str
    snippet: str
    description: str = ""

class DateExtractor:
    """Utility class for extracting and parsing dates from text"""
    
    # Comprehensive date patterns for policy updates
    POLICY_DATE_PATTERNS = [
        r'(?:updated|effective|announced|implemented|launched|revised|introduced|started|beginning)\s+(?:on\s+)?(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
        r'(?:updated|effective|announced|implemented|launched|revised|introduced|started|beginning)\s+(?:on\s+)?(\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{2,4})',
        r'(?:updated|effective|announced|implemented|launched|revised|introduced|started|beginning)\s+(?:on\s+)?(\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{2,4})',
        r'(?:from|since|as of)\s+(\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{2,4})',
        r'(?:from|since|as of)\s+(\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{2,4})',
        r'(\d{4})\s+(?:budget|policy|update|changes)',
        r'(?:in|for)\s+(\d{4})',
    ]
    
    @classmethod
    def extract_policy_dates(cls, text: str) -> List[str]:
        """Extract dates related to policies from text"""
        if not text:
            return []
        
        policy_dates = []
        for pattern in cls.POLICY_DATE_PATTERNS:
            matches = re.findall(pattern, text, flags=re.IGNORECASE)
            policy_dates.extend(matches)
        
        return list(set(policy_dates))  # Remove duplicates

class TextCleaner:
    """Utility class for cleaning and processing text"""
    
    # Patterns for removing metadata
    METADATA_PATTERNS = [
        r'^\d+\s+(hours?|days?|weeks?|months?)\s+ago\s*[·•-]\s*',
        r'^\w+\s*[·•-]\s*(?!\s*(?:updated|effective|announced|policy))',
        r'\s*[·•-]\s*\d+\s+(hours?|days?|weeks?|months?)\s+ago.*\]',
    ]
    
    # Time patterns to remove
    TIME_PATTERNS = [
        r'\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm)?\b',
        r'\b\d{1,2}\s*(?:AM|PM|am|pm)\b',
    ]
    
    # Keywords for identifying important content
    POLICY_KEYWORDS = [
        'policy', 'budget', 'government', 'change', 'support', 'economy', 
        'families', 'workers', 'jobs', 'benefit', 'insurance', 'health', 
        'cost', 'coverage', 'updated', 'effective', 'announced', 'implemented',
        'launched', 'revised', 'introduced', 'started', 'beginning', 'from',
        'since', 'as of', 'new', 'latest', 'recent', 'current'
    ]
    
    @classmethod
    def clean_text(cls, text: str) -> str:
        """Remove unwanted patterns from text"""
        if not text:
            return ""
        
        # Remove ellipses and clean up
        text = text.replace("...", "")
        
        # Remove metadata and time patterns
        for pattern in cls.TIME_PATTERNS + cls.METADATA_PATTERNS:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        return text.strip()
    
    @classmethod
    def extract_important_sentences(cls, text: str, policy_dates: List[str]) -> List[str]:
        """Extract sentences containing policy-related keywords"""
        sentences = re.split(r'(?<=[.?!])\s+', text)
        important = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            
            # Skip short or malformed sentences
            if len(sentence) < 15 or 'from .' in sentence:
                continue
            
            # Skip sentences with ellipses unless they contain policy information
            if ('...' in sentence and 
                not any(word in sentence.lower() for word in ['policy', 'updated', 'effective', 'announced'])):
                continue
            
            # Check if sentence contains policy keywords
            if any(keyword in sentence.lower() for keyword in cls.POLICY_KEYWORDS):
                # Add date context if available and sentence doesn't already have a date
                sentence_with_context = cls._add_date_context(sentence, policy_dates)
                important.append(sentence_with_context)
        
        return important
    
    @classmethod
    def _add_date_context(cls, sentence: str, policy_dates: List[str]) -> str:
        """Add date context to sentence if it doesn't already contain a date"""
        if not policy_dates:
            return sentence
        
        # Check if sentence already contains a date
        has_date = bool(re.search(r'\b\d{4}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)', 
                                 sentence, re.IGNORECASE))
        
        # Add date context for policy-related sentences without dates
        if (not has_date and 
            any(word in sentence.lower() for word in ['policy', 'benefit', 'coverage', 'support'])):
            latest_date = policy_dates[-1]
            return f"[{latest_date}] {sentence}"
        
        return sentence

def enforce_prefixes(llm_output: str) -> str:
    """Ensure all lines have date prefixes"""
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
    """Main tools class for web search and LLM integration"""
    
    def __init__(self):
        self.config = Config()
        self.date_extractor = DateExtractor()
        self.text_cleaner = TextCleaner()
        self._setup_llm()
        self._setup_tools()
    
    def _setup_llm(self) -> None:
        """Initialize LLM and embedding models"""
        try:
            self.llm = OpenAI(
                model=self.config.MODEL_NAME,
                api_key=self.config.OPENAI_API_KEY,
                system_prompt=getattr(self.config, 'system_prompt', ''),
                temperature=0.2,
                max_tokens=512,
            )
            self.embed_model = FastEmbedEmbedding()
            Settings.llm = self.llm
            Settings.embed_model = self.embed_model
            logger.info("LLM and embedding models initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise
    
    def _setup_tools(self) -> None:
        """Initialize AI tools"""
        self.search_tool = FunctionTool.from_defaults(
            fn=self.search,
            name="web_search",
            description="Search the web for current information and news"
        )
        logger.info("Search tool initialized successfully")

    def search(self, query: str) -> str:
        """
        Search Google for information related to the query.
        Returns clean, structured summaries using LLM refinement.
        
        Args:
            query: Search query string
            
        Returns:
            Formatted search results or error message
        """
        if not query or not query.strip():
            return "Empty query provided."
        
        try:
            logger.info(f"Searching for: {query}")
            
            # Perform Google search
            results = google_search(
                query,
                num_results=getattr(self.config, 'MAX_SEARCH_RESULTS', 10),
                lang=getattr(self.config, 'SEARCH_LANGUAGE', 'en'),
                advanced=True
            )

            # Process search results
            cleaned_results = self._process_search_results(results)
            
            if not cleaned_results:
                return "No relevant summaries found."

            # Generate LLM summary
            return self._generate_llm_summary(cleaned_results)

        except Exception as e:
            error_msg = f"Search failed: {str(e)}"
            logger.error(error_msg)
            return error_msg

    def _process_search_results(self, results) -> List[str]:
        """Process raw search results and extract relevant information"""
        cleaned_results = []
        
        for result in results:
            try:
                # Extract description from result object
                description = (getattr(result, 'snippet', None) or 
                             getattr(result, 'description', None))
                
                if description:
                    cleaned = self._clean_search_result(description)
                    if cleaned:
                        cleaned_results.append(cleaned)
            except Exception as e:
                logger.warning(f"Error processing search result: {e}")
                continue
        
        return cleaned_results

    def _clean_search_result(self, text: str) -> str:
        """Extract policy dates and filter relevant sentences with proper formatting"""
        if not text:
            return ""

        # Extract policy dates
        policy_dates = self.date_extractor.extract_policy_dates(text)
        
        # Clean the text
        cleaned_text = self.text_cleaner.clean_text(text)
        
        # Extract important sentences
        important_sentences = self.text_cleaner.extract_important_sentences(
            cleaned_text, policy_dates
        )

        # Build result
        result_parts = []
        
        # Add policy dates summary if found
        if policy_dates:
            unique_dates = list(set(policy_dates))
            result_parts.append(f"Policy Update Dates: {', '.join(unique_dates)}")
        
        # Add important sentences
        if important_sentences:
            result_parts.extend(important_sentences)
        
        # Join with double line breaks for better readability
        return (os.linesep + os.linesep).join(result_parts)

    def _generate_llm_summary(self, cleaned_results: List[str]) -> str:
        """Generate LLM summary from cleaned search results"""
        # Join results with clear separation
        cleaned_paragraphs = "\n\n\n".join(cleaned_results)

        # Enhanced prompt for better summarization
        prompt = (
            "Please organize the following search results into clear, factual bullet points. "
            "IMPORTANT: Preserve all policy update dates, effective dates, and timing information. "
            "Include when policies were announced, implemented, or became effective. "
            "Format each point clearly with dates in brackets when available. "
            "Avoid ellipses, vague terms, or broken phrases. "
            "Group related information together and maintain chronological order when possible:\n\n"
            + cleaned_paragraphs
        )

        try:
            response = self.llm.complete(prompt)
            return str(response)
        except Exception as e:
            logger.error(f"LLM summarization failed: {e}")
            return f"Summarization failed, returning raw results:\n\n{cleaned_paragraphs}"

    def get_search_tool(self) -> FunctionTool:
        """Return the search tool for external use"""
        return self.search_tool

    def get_config(self) -> Config:
        """Return the configuration object"""
        return self.config