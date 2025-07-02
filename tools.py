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
<<<<<<< HEAD
    """Utility class for extracting and parsing dates from text"""
=======
>>>>>>> 14465543f53d2a7906b384b3e31a4a464f9a2b6f
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
<<<<<<< HEAD
    """Utility class for cleaning and processing text"""
=======
>>>>>>> 14465543f53d2a7906b384b3e31a4a464f9a2b6f
    METADATA_PATTERNS = [
        r'^\d+\s+(hours?|days?|weeks?|months?)\s+ago\s*[·•-]\s*',
        r'^\w+\s*[·•-]\s*(?!\s*(?:updated|effective|announced|policy))',
        r'\s*[·•-]\s*\d+\s+(hours?|days?|weeks?|months?)\s+ago.*\]',
    ]
<<<<<<< HEAD
=======
    
>>>>>>> 14465543f53d2a7906b384b3e31a4a464f9a2b6f
    TIME_PATTERNS = [
        r'\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm)?\b',
        r'\b\d{1,2}\s*(?:AM|PM|am|pm)\b',
    ]
<<<<<<< HEAD
=======
    
    # Enhanced keywords for better medical/insurance content extraction
>>>>>>> 14465543f53d2a7906b384b3e31a4a464f9a2b6f
    POLICY_KEYWORDS = [
        'policy', 'budget', 'government', 'change', 'support', 'economy', 
        'families', 'workers', 'jobs', 'benefit', 'insurance', 'health', 
        'cost', 'coverage', 'updated', 'effective', 'announced', 'implemented',
        'launched', 'revised', 'introduced', 'started', 'beginning', 'from',
        'since', 'as of', 'new', 'latest', 'recent', 'current',
        # Medical/Hospital keywords
        'hospital', 'medical', 'treatment', 'stroke', 'emergency', 'panel',
        'prushield', 'pruextra', 'mount elizabeth', 'gleneagles', 'raffles',
        'claimable', 'copayment', 'cashless', 'admission', 'specialist'
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

<<<<<<< HEAD
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
=======
class Tools:
>>>>>>> 14465543f53d2a7906b384b3e31a4a464f9a2b6f
    def __init__(self):
        self.config = Config()
        self.date_extractor = DateExtractor()
        self.text_cleaner = TextCleaner()
        self._setup_llm()
        self._setup_tools()
<<<<<<< HEAD
=======
        self._setup_rag(pdf_dir=getattr(self.config, "PDF_DIR", "docs"))

>>>>>>> 14465543f53d2a7906b384b3e31a4a464f9a2b6f
    def _setup_llm(self) -> None:
        try:
            self.llm = OpenAI(
                model=self.config.MODEL_NAME,
                api_key=self.config.OPENAI_API_KEY,
                system_prompt=getattr(self.config, 'system_prompt', ''),
                temperature=0.1,  # Lower temperature for more consistent formatting
                max_tokens=1024,  # Increased for longer structured responses
            )
            self.embed_model = FastEmbedEmbedding()
            Settings.llm = self.llm
            Settings.embed_model = self.embed_model
            logger.info("LLM and embedding models initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise
<<<<<<< HEAD
=======

>>>>>>> 14465543f53d2a7906b384b3e31a4a464f9a2b6f
    def _setup_tools(self) -> None:
        self.search_tool = FunctionTool.from_defaults(
            fn=self.query_with_rag_then_search,
            name="hybrid_search",
            description="Answer questions using PDF knowledge base first, then fallback to web search"
        )
        self.hospital_claim_tool = FunctionTool.from_defaults(
            fn=self.assess_hospital_claim,
            name="hospital_claim_assessment",
            description="Assess hospitalization claim eligibility and co-payment for PRUShield + PRUExtra"
        )
        logger.info("Search and hospital claim tools initialized successfully")

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

        # Enhanced prompt engineering for medical/insurance queries
        medical_keywords = ["stroke", "heart attack", "emergency", "hospital", "claimable", "prushield", "copayment", "panel", "treatment"]
        
        # Detect query type and enhance accordingly
        if any(kw in query.lower() for kw in medical_keywords):
            enhanced_query = f"{query} Singapore PRUShield panel hospitals Mount Elizabeth Gleneagles Mount Alvernia coverage network"
        else:
            enhanced_query = query

        # Step 1: Try RAG first
        if self.rag_engine:
            try:
                rag_response = self.rag_engine.query(enhanced_query)
                if rag_response and len(str(rag_response).strip()) > 50:
                    return f"📄 **Answer from PDF knowledge base:**\n\n{str(rag_response).strip()}"
            except Exception as e:
                logger.warning(f"RAG query failed: {e}")

        # Step 2: Enhanced web search with structured output
        return self.search(enhanced_query)

    def search(self, query: str) -> str:
<<<<<<< HEAD
        if not query or not query.strip():
            return "Empty query provided."
=======
>>>>>>> 14465543f53d2a7906b384b3e31a4a464f9a2b6f
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
<<<<<<< HEAD
                return "No relevant summaries found."
            return self._generate_llm_summary(cleaned_results)
=======
                return "No relevant information found."
            return self._generate_structured_response(cleaned_results, query)
>>>>>>> 14465543f53d2a7906b384b3e31a4a464f9a2b6f
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            return f"Search failed: {str(e)}"

<<<<<<< HEAD
    def _process_search_results(self, results) -> List[str]:
        cleaned_results = []
        for result in results:
            try:
                description = (getattr(result, 'snippet', None) or 
                             getattr(result, 'description', None))
=======
    def _process_search_results(self, results) -> List[Dict[str, str]]:
        """Enhanced result processing with metadata preservation"""
        cleaned_results = []
        for result in results:
            try:
                title = getattr(result, 'title', '')
                url = getattr(result, 'url', '')
                description = (getattr(result, 'snippet', None) or 
                               getattr(result, 'description', None) or '')
                
>>>>>>> 14465543f53d2a7906b384b3e31a4a464f9a2b6f
                if description:
                    cleaned_content = self._clean_search_result(description)
                    if cleaned_content:
                        cleaned_results.append({
                            'title': title,
                            'url': url,
                            'content': cleaned_content,
                            'source': self._extract_source_name(title, url)
                        })
            except Exception as e:
                logger.warning(f"Error processing search result: {e}")
<<<<<<< HEAD
                continue
=======
>>>>>>> 14465543f53d2a7906b384b3e31a4a464f9a2b6f
        return cleaned_results

    def _extract_source_name(self, title: str, url: str) -> str:
        """Extract clean source name from title or URL"""
        if 'gleneagles' in title.lower() or 'gleneagles' in url.lower():
            return 'Gleneagles Hospital'
        elif 'mount elizabeth' in title.lower() or 'mountelizabeth' in url.lower():
            return 'Mount Elizabeth Hospital'
        elif 'raffles' in title.lower() or 'raffles' in url.lower():
            return 'Raffles Medical'
        elif 'prudential' in title.lower() or 'prudential' in url.lower():
            return 'Prudential Singapore'
        elif 'moh' in title.lower() or 'moh' in url.lower():
            return 'Ministry of Health'
        else:
            # Extract domain name as fallback
            try:
                from urllib.parse import urlparse
                domain = urlparse(url).netloc
                return domain.replace('www.', '').title()
            except:
                return 'Healthcare Provider'

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
<<<<<<< HEAD
            unique_dates = list(set(policy_dates))
            result_parts.append(f"Policy Update Dates: {', '.join(unique_dates)}")
=======
            result_parts.append(f"Policy Update Dates: {', '.join(policy_dates)}")
>>>>>>> 14465543f53d2a7906b384b3e31a4a464f9a2b6f
        if important_sentences:
            result_parts.extend(important_sentences)
        return (os.linesep + os.linesep).join(result_parts)

<<<<<<< HEAD
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
=======
    def _generate_structured_response(self, cleaned_results: List[Dict[str, str]], original_query: str) -> str:
        """Generate structured response using enhanced prompting"""
        
        # Prepare structured content for LLM
        formatted_content = []
        for i, result in enumerate(cleaned_results):
            source_info = f"Source: {result['source']}"
            content_block = f"**{result['title']}**\n{source_info}\n{result['content']}\n"
            formatted_content.append(content_block)
        
        # Enhanced prompt for structured output
        prompt = f"""
Based on the user query: "{original_query}"

Organize the following search results into a comprehensive, structured response following these requirements:

1. Start with an appropriate emoji header (🏥 for medical, 💊 for medication, 📋 for policy)
2. Use markdown formatting with **bold headers**
3. Structure the response with clear sections:
   - Hospital Panel Options (if medical query - MUST mention hospitals within Prudential's panel network)
   - Coverage Benefits 
   - Important Considerations
   - Next Steps (always include contact info: 1800 333 0333)

4. Include specific details like:
   - Hospital names that are within Prudential's panel (Mount Elizabeth, Gleneagles, Mount Alvernia, etc.)
   - Clearly distinguish between panel vs non-panel hospitals
   - Coverage percentages for panel hospitals
   - Prudential product names (PRUShield, PRUExtra, PRUPanel Connect)
   - Actionable advice

5. CRITICAL: Always mention which hospitals are within Prudential's panel network and their coverage benefits
6. Maintain professional, confident tone positioning Prudential as the preferred choice

Search Results:
{chr(10).join(formatted_content)}

Generate a response that directly addresses the user's query with the structured format above.
"""

>>>>>>> 14465543f53d2a7906b384b3e31a4a464f9a2b6f
        try:
            response = self.llm.complete(prompt)
            return str(response)
        except Exception as e:
            logger.error(f"LLM structured response failed: {e}")
            # Fallback to basic formatting
            return self._create_fallback_response(cleaned_results, original_query)

    def _create_fallback_response(self, cleaned_results: List[Dict[str, str]], query: str) -> str:
        """Fallback response if LLM fails"""
        response = "## **🏥 Search Results**\n\n"
        
        for result in cleaned_results[:5]:  # Limit to top 5 results
            response += f"### **{result['title']}**\n"
            response += f"*Source: {result['source']}*\n\n"
            response += f"{result['content']}\n\n"
        
        response += "### **Next Steps**\n"
        response += "Contact Prudential Customer Service at **1800 333 0333** for personalized assistance.\n"
        
        return response

<<<<<<< HEAD
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
=======
    def get_search_tool(self) -> FunctionTool:
        return self.search_tool
>>>>>>> 14465543f53d2a7906b384b3e31a4a464f9a2b6f

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