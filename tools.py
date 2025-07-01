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
    title: str
    url: str
    snippet: str
    description: str = ""

class DateExtractor:
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
        if not text:
            return []
        policy_dates = []
        for pattern in cls.POLICY_DATE_PATTERNS:
            matches = re.findall(pattern, text, flags=re.IGNORECASE)
            policy_dates.extend(matches)
        return list(set(policy_dates))

class TextCleaner:
    METADATA_PATTERNS = [
        r'^\d+\s+(hours?|days?|weeks?|months?)\s+ago\s*[·•-]\s*',
        r'^\w+\s*[·•-]\s*(?!\s*(?:updated|effective|announced|policy))',
        r'\s*[·•-]\s*\d+\s+(hours?|days?|weeks?|months?)\s+ago.*\]',
    ]
    
    TIME_PATTERNS = [
        r'\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm)?\b',
        r'\b\d{1,2}\s*(?:AM|PM|am|pm)\b',
    ]
    
    POLICY_KEYWORDS = [
        'policy', 'budget', 'government', 'change', 'support', 'economy', 
        'families', 'workers', 'jobs', 'benefit', 'insurance', 'health', 
        'cost', 'coverage', 'updated', 'effective', 'announced', 'implemented',
        'launched', 'revised', 'introduced', 'started', 'beginning', 'from',
        'since', 'as of', 'new', 'latest', 'recent', 'current'
    ]
    
    @classmethod
    def clean_text(cls, text: str) -> str:
        if not text:
            return ""
        text = text.replace("...", "")
        for pattern in cls.TIME_PATTERNS + cls.METADATA_PATTERNS:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        return text.strip()
    
    @classmethod
    def extract_important_sentences(cls, text: str, policy_dates: List[str]) -> List[str]:
        sentences = re.split(r'(?<=[.?!])\s+', text)
        important = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 15 or 'from .' in sentence:
                continue
            if ('...' in sentence and 
                not any(word in sentence.lower() for word in ['policy', 'updated', 'effective', 'announced'])):
                continue
            if any(keyword in sentence.lower() for keyword in cls.POLICY_KEYWORDS):
                sentence_with_context = cls._add_date_context(sentence, policy_dates)
                important.append(sentence_with_context)
        return important
    
    @classmethod
    def _add_date_context(cls, sentence: str, policy_dates: List[str]) -> str:
        if not policy_dates:
            return sentence
        has_date = bool(re.search(r'\b\d{4}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)', 
                                 sentence, re.IGNORECASE))
        if (not has_date and 
            any(word in sentence.lower() for word in ['policy', 'benefit', 'coverage', 'support'])):
            latest_date = policy_dates[-1]
            return f"[{latest_date}] {sentence}"
        return sentence

class Tools:
    def __init__(self):
        self.config = Config()
        self.date_extractor = DateExtractor()
        self.text_cleaner = TextCleaner()
        self._setup_llm()
        self._setup_tools()
        self._setup_rag(pdf_dir=getattr(self.config, "PDF_DIR", "docs"))

    def _setup_llm(self) -> None:
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
        self.search_tool = FunctionTool.from_defaults(
            fn=self.query_with_rag_then_search,
            name="hybrid_search",
            description="Answer questions using PDF knowledge base first, then fallback to web search"
        )
        logger.info("Search tool initialized successfully")

    def _setup_rag(self, pdf_dir: str) -> None:
        try:
            from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, ServiceContext
            from llama_index.core.query_engine import RetrieverQueryEngine

            documents = SimpleDirectoryReader(pdf_dir).load_data()
            service_context = ServiceContext.from_defaults(
                llm=self.llm,
                embed_model=self.embed_model
            )
            self.index = VectorStoreIndex.from_documents(documents, service_context=service_context)
            self.rag_engine = self.index.as_query_engine(similarity_top_k=3)
            logger.info("RAG engine initialized from directory: %s", pdf_dir)
        except Exception as e:
            logger.error(f"Failed to initialize RAG engine: {e}")
            self.rag_engine = None

    def query_with_rag_then_search(self, query: str) -> str:
        """Use RAG engine first, fallback to web search if needed"""
        if not query.strip():
            return "Please enter a valid query."

    def query_with_rag_then_search(self, query: str) -> str:
        """Use RAG engine first, fallback to web search if needed"""

        if not query.strip():
            return "Please enter a valid query."

        # --- Prompt enhancement for medical/insurance queries ---
        medical_keywords = ["stroke", "heart attack", "emergency", "hospital", "claimable", "prushield", "copayment"]
        if any(kw in query.lower() for kw in medical_keywords):
            query += (
                " Based on Singapore MOH and Prudential PRUShield policies, "
                "please specify which hospital panel the patient should go to for stroke treatment, "
                "whether it is claimable by PRUShield, and provide estimated co-payment amounts "
                "according to the latest regulations."
            )

        # Step 1: Try RAG
        if self.rag_engine:
            try:
                rag_response = self.rag_engine.query(query)
                if rag_response and len(str(rag_response).strip()) > 50:
                    return f"📄 Answer from PDF knowledge base:\n\n{str(rag_response).strip()}"
            except Exception as e:
                logger.warning(f"RAG query failed: {e}")

        # Step 2: Fallback to Search
        return f"🌐 Fallback to web search:\n\n{self.search(query)}"

    def search(self, query: str) -> str:
        try:
            logger.info(f"Searching for: {query}")
            results = google_search(
                query,
                num_results=getattr(self.config, 'MAX_SEARCH_RESULTS', 10),
                lang=getattr(self.config, 'SEARCH_LANGUAGE', 'en'),
                advanced=True
            )
            cleaned_results = self._process_search_results(results)
            if not cleaned_results:
                return "No relevant summaries found."
            return self._generate_llm_summary(cleaned_results)
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            return f"Search failed: {str(e)}"

    def _process_search_results(self, results) -> List[str]:
        cleaned_results = []
        for result in results:
            try:
                description = (getattr(result, 'snippet', None) or 
                               getattr(result, 'description', None))
                if description:
                    cleaned = self._clean_search_result(description)
                    if cleaned:
                        cleaned_results.append(cleaned)
            except Exception as e:
                logger.warning(f"Error processing search result: {e}")
        return cleaned_results

    def _clean_search_result(self, text: str) -> str:
        if not text:
            return ""
        policy_dates = self.date_extractor.extract_policy_dates(text)
        cleaned_text = self.text_cleaner.clean_text(text)
        important_sentences = self.text_cleaner.extract_important_sentences(
            cleaned_text, policy_dates
        )
        result_parts = []
        if policy_dates:
            result_parts.append(f"Policy Update Dates: {', '.join(policy_dates)}")
        if important_sentences:
            result_parts.extend(important_sentences)
        return (os.linesep + os.linesep).join(result_parts)

    def _generate_llm_summary(self, cleaned_results: List[str]) -> str:
        prompt = (
            "Organize the following search results into clear, factual bullet points. "
            "Preserve all policy update dates and timing information. "
            "Group related info and keep it chronological:\n\n"
            + "\n\n\n".join(cleaned_results)
        )
        try:
            response = self.llm.complete(prompt)
            return str(response)
        except Exception as e:
            logger.error(f"LLM summarization failed: {e}")
            return f"Summarization failed. Raw results:\n\n{cleaned_results}"

    def get_search_tool(self) -> FunctionTool:
        return self.search_tool

    def get_config(self) -> Config:
        return self.config
