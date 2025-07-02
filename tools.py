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
    """Utility class for cleaning and processing text"""
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

def enforce_prefixes(llm_output: str) -> str:
    date_pattern = re.compile(r"^\(\d{4}(?:-\d{2}-\d{2})?\)")
    lines = [line.strip() for line in llm_output.split('\n') if line.strip()]
    processed = []
    for line in lines:
        if date_pattern.match(line):
            processed.append(line)
        else:
            processed.append(f"(Date not specified) {line}")
    return "\n".join(processed)

# --- INSURANCE CLAIM HOSPITAL PANEL LOGIC ---
HOSPITAL_PANEL = {
    "Tan Tock Seng": {
        "type": "Public",
        "panel": True,
        "deductible": "S$2,000",
        "copay": "5% capped at S$3,000/year",
    },
    "Mount Elizabeth": {
        "type": "Private",
        "panel": True,
        "deductible": "S$3,500",
        "copay": "5% capped at S$3,000/year",
    },
    "Gleneagles": {
        "type": "Private",
        "panel": True,
        "deductible": "S$3,500",
        "copay": "5% capped at S$3,000/year",
    },
    "Singapore General Hospital": {
        "type": "Public",
        "panel": True,
        "deductible": "S$2,000",
        "copay": "5% capped at S$3,000/year",
    },
    # Expand this dictionary as needed for more hospitals
}
DEFAULT_RULES = {
    "non_panel": {
        "deductible": "Varies",
        "copay": "10-20% (usually uncapped)",
    }
}

# --- PREDEFINED ANSWERS DICTIONARY ---
PREDEFINED_ANSWERS = {
    "my dad had a stroke, should i go to tan tock seng or mount e?": 
    """
**Tan Tock Seng Hospital:**
- Type: Public hospital  
- Panel status: Panel  
- Claimable under PRUShield + PRUExtra: Yes (if you use a panel specialist)  
- Estimated deductible: S$2,000  
- Estimated co-payment: 5% capped at S$3,000/year  

**Mount Elizabeth Hospital:**
- Type: Private hospital  
- Panel status: Panel  
- Claimable under PRUShield + PRUExtra: Yes (if you use a panel specialist)  
- Estimated deductible: S$3,500  
- Estimated co-payment: 5% capped at S$3,000/year  

**Notes:**
- Emergency admissions are generally covered regardless of panel status.
- Always check if your specialist is on the panel for best benefits.
- Non-panel admissions may incur higher co-payment and deductibles.
"""
    # Add more Q&A as needed
}

class Tools:
    """Main tools class for web search, LLM integration, and hospital claim assessment"""
    def __init__(self):
        self.config = Config()
        self.date_extractor = DateExtractor()
        self.text_cleaner = TextCleaner()
        self._setup_llm()
        self._setup_tools()
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
            fn=self.search,
            name="web_search",
            description="Search the web for current information and news"
        )
        self.hospital_claim_tool = FunctionTool.from_defaults(
            fn=self.assess_hospital_claim,
            name="hospital_claim_assessment",
            description="Assess hospitalization claim eligibility and co-payment for PRUShield + PRUExtra"
        )
        logger.info("Search and hospital claim tools initialized successfully")

    def search(self, query: str) -> str:
        if not query or not query.strip():
            return "Empty query provided."
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
            error_msg = f"Search failed: {str(e)}"
            logger.error(error_msg)
            return error_msg

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
                continue
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
            unique_dates = list(set(policy_dates))
            result_parts.append(f"Policy Update Dates: {', '.join(unique_dates)}")
        if important_sentences:
            result_parts.extend(important_sentences)
        return (os.linesep + os.linesep).join(result_parts)

    def _generate_llm_summary(self, cleaned_results: List[str]) -> str:
        cleaned_paragraphs = "\n\n\n".join(cleaned_results)
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

    # --- INSURANCE AGENTIC TOOL ---
    def assess_hospital_claim(self, user_query: str) -> str:
        """
        Given a user's hospital/situation query, answer:
        1. Which panel hospital(s) they can go to
        2. Whether it’s likely claimable under their PRUShield + PRUExtra plan
        3. Estimated co-payment based on current regulations
        """
        # Extract mentioned hospitals (basic substring match; for prod use NLP/NER)
        hospital_matches = []
        for h in HOSPITAL_PANEL.keys():
            if re.search(re.escape(h.lower()), user_query.lower()):
                hospital_matches.append(h)
        # If none found, check for substrings (partial match)
        if not hospital_matches:
            for h in HOSPITAL_PANEL.keys():
                for word in user_query.lower().split():
                    if word in h.lower():
                        hospital_matches.append(h)
        # If still none, provide a helpful error
        if not hospital_matches:
            return (
                "I couldn't identify a hospital in your message. "
                "Please specify which hospital(s) you are considering (e.g. 'Tan Tock Seng', 'Mount Elizabeth')."
            )

        resp = []
        for hosp in set(hospital_matches):
            info = HOSPITAL_PANEL.get(hosp, {})
            panel = "Panel" if info.get("panel") else "Non-panel"
            hosp_type = info.get("type", "Unknown")
            deductible = info.get("deductible", DEFAULT_RULES['non_panel']['deductible'])
            copay = info.get("copay", DEFAULT_RULES['non_panel']['copay'])
            resp.append(
                f"**{hosp}:**\n"
                f"- Type: {hosp_type} hospital\n"
                f"- Panel status: {panel}\n"
                f"- Claimable under PRUShield + PRUExtra: {'Yes (if you use a panel specialist)' if info.get('panel') else 'Possible, but less favorable terms'}\n"
                f"- Estimated deductible: {deductible}\n"
                f"- Estimated co-payment: {copay}\n"
            )
        resp.append(
            "Notes:\n"
            "- Emergency admissions are generally covered regardless of panel status.\n"
            "- Always check if your specialist is on the panel for best benefits.\n"
            "- Non-panel admissions may incur higher co-payment and deductibles.\n"
        )
        return "\n".join(resp)

    # --- MAIN QUERY ROUTER ---
    def answer(self, user_query: str) -> str:
        """
        Main entry point: Checks for predefined answers, then insurance logic, then web search.
        """
        normalized_query = user_query.strip().lower()
        # 1. Check for exact predefined answer
        if normalized_query in PREDEFINED_ANSWERS:
            return PREDEFINED_ANSWERS[normalized_query]
        # 2. If a hospital name is mentioned, or claim/insurance-related term, use insurance logic
        for h in HOSPITAL_PANEL.keys():
            if h.lower() in normalized_query:
                return self.assess_hospital_claim(user_query)
        if any(term in normalized_query for term in [
            "prushield", "pruextra", "panel hospital", "is it claimable", "co-payment", "deductible"
        ]):
            return self.assess_hospital_claim(user_query)
        # 3. Otherwise, default to web search
        return self.search(user_query)

    def get_search_tool(self) -> FunctionTool:
        return self.search_tool
    def get_hospital_claim_tool(self) -> FunctionTool:
        return self.hospital_claim_tool
    def get_config(self) -> Config:
        return self.config