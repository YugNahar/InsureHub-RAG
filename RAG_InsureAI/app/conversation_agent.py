# conversation_agent.py
"""
Smart, human‑like insurance advisor.
Implements intent detection and response logic as per the specification.
"""
import asyncio
import logging
from typing import Dict, List, Tuple, Optional
import re

logger = logging.getLogger(__name__)

# States for multi‑turn
STATE_DISCOVERY = "discovery"
STATE_REFINEMENT = "refinement"
STATE_DETAILS = "details"

# ─────────────────────────────────────────────────────────────────────────────
# INSURANCE CATEGORIES & SUB‑OPTIONS (ONLY THESE)
# ─────────────────────────────────────────────────────────────────────────────
INSURANCE_CATEGORIES = {
    "motor": {
        "label": "Motor",
        "options": [
            {"id": "car", "label": "Car Insurance", "description": "Private car cover – damage, theft, third‑party liability", "recommended": True},
            {"id": "bike", "label": "Bike Insurance", "description": "For motorcycles and scooters", "recommended": False},
            {"id": "commercial", "label": "Commercial Vehicle", "description": "For trucks, taxis, business vehicles", "recommended": False}
        ]
    },
    "health": {
        "label": "Health",
        "options": [
            {"id": "individual", "label": "Individual", "description": "Personal medical coverage", "recommended": True},
            {"id": "family_floater", "label": "Family Floater", "description": "Single sum insured for entire family", "recommended": False},
            {"id": "senior", "label": "Senior Citizen", "description": "Health cover for elders", "recommended": False},
            {"id": "critical", "label": "Critical Illness", "description": "Lump sum on diagnosis of serious illnesses", "recommended": False}
        ]
    },
    "life": {
        "label": "Life",
        "options": [
            {"id": "term", "label": "Term", "description": "Pure protection – high cover at low cost", "recommended": True},
            {"id": "whole", "label": "Whole Life", "description": "Coverage for entire lifetime", "recommended": False},
            {"id": "ulip", "label": "ULIP", "description": "Investment + insurance", "recommended": False},
            {"id": "endowment", "label": "Endowment", "description": "Savings + protection", "recommended": False}
        ]
    },
    "travel": {
        "label": "Travel",
        "options": [
            {"id": "international", "label": "International", "description": "Medical & trip protection abroad", "recommended": True},
            {"id": "domestic", "label": "Domestic", "description": "For trips within your country", "recommended": False},
            {"id": "student", "label": "Student", "description": "For students studying abroad", "recommended": False},
            {"id": "baggage", "label": "Baggage Loss", "description": "Reimbursement for lost/delayed luggage", "recommended": False},
            {"id": "flight_delay", "label": "Flight Delay", "description": "Compensation for delayed flights", "recommended": False}
        ]
    },
    "home": {
        "label": "Home",
        "options": [
            {"id": "home", "label": "Home Insurance", "description": "Protects building and contents", "recommended": True},
            {"id": "fire", "label": "Fire Insurance", "description": "Cover against fire damage", "recommended": False},
            {"id": "property", "label": "Property Insurance", "description": "For rental/commercial properties", "recommended": False}
        ]
    },
    "personal": {
        "label": "Personal",
        "options": [
            {"id": "accident", "label": "Personal Accident", "description": "Compensation for accidental death/disability", "recommended": True},
            {"id": "disability", "label": "Disability", "description": "Income replacement if unable to work", "recommended": False}
        ]
    }
}

# Follow‑up questions after showing options
FOLLOW_UP_QUESTIONS = {
    "motor": "Which vehicle type do you need coverage for?",
    "health": "Are you looking for individual or family coverage?",
    "life": "Do you want pure protection or an investment‑linked plan?",
    "travel": "Where are you travelling? (Domestic or International)",
    "home": "Is this for your own home or a rental property?",
    "personal": "Is this for accidental injury or long‑term disability?"
}

class ConversationAgent:
    # Keep at most this many user/assistant turn pairs in session history
    _MAX_HISTORY_TURNS = 5

    def __init__(self, vector_store, multi_rag):
        self.vector_store = vector_store
        self.multi_rag = multi_rag
        self.sessions: Dict[str, dict] = {}

    # -------------------------------------------------------------------------
    # STEP 1: INTENT DETECTION
    # -------------------------------------------------------------------------
    def _classify_intent(self, question: str) -> str:
        """
        Returns:
            "SMALL_TALK"  – plain greetings, thanks, casual chat with no real request
            "DISCOVERY"   – user wants to explore/buy a specific category
            "CATEGORY"    – user directly names a category (e.g., "Home insurance")
            "GENERAL"     – user asks for all insurance options
            "INFORMATIONAL" – user asks a specific question
        """
        q = question.lower().strip()

        # ── KNOWLEDGE_BASE / CAPABILITY: user asking about what the assistant knows or has access to ──
        knowledge_base_phrases = [
            "what do you know", "what all have you ingested", "what have you ingested",
            "what documents", "what is in your knowledge base", "what is your knowledge base",
            "what information", "what data", "what can you do",
            "how do you work", "how do you know", "how does this work",
            "tell me about yourself", "what are you", "what are your capabilities",
            "what sources", "what policies", "what insurance",
            "knowledge base", "what have you been trained on", "what are you trained on",
            "what files", "what can you help with", "what can i ask",
        ]
        if any(phrase in q for phrase in knowledge_base_phrases):
            return "KNOWLEDGE_BASE"

        # ── SMALL_TALK: pure greetings / thanks / casual chat (no embedded question or request) ──
        _PURE_GREETINGS = {
            # Basic greetings
            "hi", "hello", "hey", "hiya", "heya", "yo", "sup", "howdy",
            "hi there", "hello there", "hey there", "hiya there",
            "hi hi", "hello hello", "hey hey",
            # Time-based greetings
            "good morning", "good afternoon", "good evening", "good night",
            "gm", "gn", "good day", "good to see you",
            # How are you variants
            "how are you", "how are you doing", "how are you today",
            "how's it going", "how is it going", "how's everything",
            "how are things", "how are things going", "how do you do",
            "how have you been", "how's your day", "how is your day",
            "how's your day going", "you doing okay", "you good",
            "all good", "hope you're well", "hope you are well",
            # Thanks
            "thanks", "thank you", "thank you so much", "thanks a lot",
            "thanks so much", "many thanks", "thanks a ton", "ty",
            "thx", "thnx", "thnks", "cheers", "much appreciated",
            "appreciate it", "appreciate that", "thanks for that",
            "thank you very much", "thank u", "thanks mate",
            # Farewells
            "bye", "goodbye", "good bye", "bye bye", "later", "see ya",
            "see you", "see you later", "see you soon", "take care",
            "have a good day", "have a great day", "have a nice day",
            "have a good one", "catch you later", "talk later",
            "talk soon", "cya", "ttyl", "peace", "adios", "hasta la vista",
            # Acknowledgements
            "ok", "okay", "ok thanks", "okay thanks", "alright", "cool",
            "got it", "sounds good", "perfect", "great", "nice", "noted",
            "understood", "makes sense", "sure", "yep", "yup", "yeah",
            "no problem", "no worries", "sure thing", "of course",
            # Greetings with name
            "hi insureai", "hello insureai", "hey insureai", "hi layla",
            "hello layla", "hey layla",
            # Restart / other
            "start", "start over", "restart", "reset", "menu", "help",
            "hi again", "hello again", "hey again", "back again",
            "i'm back", "im back",
        }
        stripped = q.strip().strip("!.,? ").strip()
        # Check if the message is exactly a small-talk phrase (no extra content)
        if stripped in _PURE_GREETINGS:
            return "SMALL_TALK"
        # Also match if the message has only small-talk words separated by punctuation/whitespace
        # (e.g. "hi!" , "hello!!" , "good morning!" , "hey there" , "thank you!")
        words = re.findall(r"[a-zA-Z'!.,?]+", stripped)
        if words:
            combined = " ".join(w.strip("!.,? ").lower() for w in words if w.strip("!.,? ")).strip()
            if combined and combined in _PURE_GREETINGS:
                return "SMALL_TALK"
        # ── END SMALL_TALK ──

        # GENERAL intent – wants to see all options
        general_phrases = [
            "what insurance", "what options", "all insurance", "list insurance",
            "what do you have", "show me all", "insurance types", "categories"
        ]
        if any(phrase in q for phrase in general_phrases):
            return "GENERAL"
        
        # DISCOVERY intent – wants to buy/explore a specific category
        discovery_patterns = [
            r"suggest (?:me )?(?:a )?(?:car|bike|travel|health|life|home) insurance",
            r"recommend (?:me )?(?:a )?(?:car|bike|travel|health|life|home) insurance",
            r"want (?:to )?(?:buy|purchase|get|take) (?:a )?(?:car|bike|travel|health|life|home) insurance",
            r"need (?:to )?(?:buy|purchase|get|take) (?:a )?(?:car|bike|travel|health|life|home) insurance",
            r"looking (?:for|to (?:buy|purchase|get|take)) (?:a )?(?:car|bike|travel|health|life|home) insurance",
            r"planning to (?:buy|purchase|get|take) (?:a )?(?:car|bike|travel|health|life|home) insurance",
            r"i want (?:a )?(?:car|bike|travel|health|life|home) insurance",
            r"i need (?:a )?(?:car|bike|travel|health|life|home) insurance",
            r"buy (?:a )?(?:car|bike|travel|health|life|home) insurance",
            r"get (?:a )?(?:car|bike|travel|health|life|home) insurance",
            r"which (?:policy|insurance) (?:should|can|shall|must|do) i (?:take|buy|get|purchase|choose|pick|select|opt)",
            r"which insurance (?:should|shall|can|must|do) i",
            r"what insurance (?:should|shall|can|must|do) i",
            r"best (?:policy|insurance)",
            r"planning to (?:buy|purchase|get) (?:a )?(?:new )?(?:car|bike|vehicle|two.?wheeler|house|home|flat)",
            r"(?:just|recently) (?:bought|purchased|got) (?:a )?(?:new )?(?:car|bike|vehicle|house|home)",
            r"(?:should|shall|can|must) i (?:buy|get|take|purchase) (?:a )?(?:car|bike|travel|health|life|home) insurance",
        ]
        for pattern in discovery_patterns:
            if re.search(pattern, q):
                return "DISCOVERY"
        
        # CATEGORY intent – user directly says a category name (short)
        category_names = ["car", "bike", "motor", "health", "life", "travel", "home", "personal"]
        if any(cat in q for cat in category_names) and len(q.split()) <= 4:
            return "CATEGORY"
        
        # SELECTION of a sub‑option (handled in refinement stage, but for first message)
        sub_options = ["international", "domestic", "student", "baggage", "flight delay",
                       "individual", "family floater", "senior", "critical",
                       "term", "whole life", "ulip", "endowment",
                       "fire", "property", "accident", "disability"]
        if any(opt in q for opt in sub_options) and len(q.split()) <= 5:
            return "CATEGORY"
        
        # Default: INFORMATIONAL
        return "INFORMATIONAL"

    def _extract_category_from_query(self, question: str) -> Tuple[str, str]:
        """
        Returns (category_id, category_label) if a category is mentioned.
        """
        q = question.lower()
        if any(w in q for w in ["car", "bike", "motor", "vehicle", "auto", "commercial vehicle"]):
            return "motor", "Motor"
        if any(w in q for w in ["health", "medical", "hospital", "surgery", "critical", "family floater", "senior"]):
            return "health", "Health"
        if any(w in q for w in ["life", "term", "whole life", "ulip", "endowment"]):
            return "life", "Life"
        if any(w in q for w in ["travel", "trip", "flight", "baggage", "delay", "international", "domestic", "student"]):
            return "travel", "Travel"
        if any(w in q for w in ["home", "house", "property", "fire"]):
            return "home", "Home"
        if any(w in q for w in ["accident", "disability", "personal"]):
            return "personal", "Personal"
        return "unclear", "General"

    def _get_all_main_categories(self) -> List[dict]:
        """Return all main categories (no sub‑options)."""
        return [
            {"id": "motor", "label": "Motor Insurance", "description": "Car, bike, commercial vehicles", "recommended": False},
            {"id": "health", "label": "Health Insurance", "description": "Individual, family floater, senior, critical illness", "recommended": False},
            {"id": "life", "label": "Life Insurance", "description": "Term, whole life, ULIP, endowment", "recommended": False},
            {"id": "travel", "label": "Travel Insurance", "description": "International, domestic, student, baggage, flight delay", "recommended": False},
            {"id": "home", "label": "Home Insurance", "description": "Home, fire, property", "recommended": False},
            {"id": "personal", "label": "Personal Insurance", "description": "Accident, disability", "recommended": False}
        ]

    def _get_sub_options(self, category_id: str) -> List[dict]:
        """Return sub‑options for a given category."""
        cat = INSURANCE_CATEGORIES.get(category_id)
        if not cat:
            return []
        return [opt for opt in cat["options"]]

    # -------------------------------------------------------------------------
    # CONVERSATION HISTORY HELPERS
    # -------------------------------------------------------------------------
    def _build_history_string(self, session_id: str) -> str:
        """Build a formatted history string from stored session messages."""
        session = self.sessions.get(session_id, {})
        messages = session.get("messages", [])
        # Take only the last N turns
        recent = messages[-self._MAX_HISTORY_TURNS * 2:]
        parts = []
        for role, content in recent:
            label = "User" if role == "user" else "Assistant"
            parts.append(f"{label}: {content}")
        return "\n".join(parts)

    def _append_turn(self, session_id: str, user_msg: str, assistant_msg: str) -> None:
        """Append a user/assistant turn to session history, capped at MAX_HISTORY_TURNS."""
        session = self.sessions.setdefault(session_id, {})
        messages = session.setdefault("messages", [])
        messages.append(("user", user_msg))
        messages.append(("assistant", assistant_msg))
        # Keep only the most recent N turns
        if len(messages) > self._MAX_HISTORY_TURNS * 2:
            session["messages"] = messages[-(self._MAX_HISTORY_TURNS * 2):]

    # -------------------------------------------------------------------------
    # DOCUMENT NAME EXTRACTION (for informational queries)
    # -------------------------------------------------------------------------
    # Common insurance/travel words that appear in document names but are too
    # generic to count as an explicit document reference in a query.
    _GENERIC_DOC_WORDS = {
        "insurance", "policy", "travel", "health", "motor", "life", "home",
        "cover", "plan", "smart", "care", "plus", "premier", "basic",
        "standard", "gold", "silver", "platinum", "group", "individual",
        "family", "personal", "general", "global", "international", "domestic",
        "the", "and", "for", "with", "from", "inbound", "outbound",
    }

    def _extract_document_names(self, question: str) -> List[str]:
        """
        Returns document names that the user is EXPLICITLY referencing.
        Matches only when:
          - the full cleaned source name appears verbatim in the query, OR
          - a word from the source name that is long (≥7 chars) AND not a
            generic insurance term appears in the query.
        This prevents common words like "travel", "home", "car" from falsely
        triggering a document filter on general questions.
        """
        all_sources = self.vector_store.list_filenames() if hasattr(self.vector_store, "list_filenames") else (self.vector_store.list_sources() if self.vector_store else [])
        if not all_sources:
            return []
        q_lower = question.lower()
        matched = []
        for src in all_sources:
            src_lower = src.lower()
            clean_src = re.sub(r'\.[a-z]+$', '', src_lower)
            # Full name match (most reliable)
            if clean_src in q_lower:
                matched.append(src)
                continue
            # Word-level match — only for long, specific words
            words = re.split(r'[_\s\-]+', clean_src)
            for word in words:
                if len(word) >= 7 and word not in self._GENERIC_DOC_WORDS and word in q_lower:
                    matched.append(src)
                    break
        return list(dict.fromkeys(matched))

    # -------------------------------------------------------------------------
    # MAIN PROCESSING
    # -------------------------------------------------------------------------
    async def process_message(
        self, session_id: str, user_message: str, history: str
    ) -> Tuple[dict, bool]:
        session = self.sessions.get(session_id, {"stage": STATE_DISCOVERY, "selected_category": None})
        stage = session.get("stage", STATE_DISCOVERY)
        selected_cat = session.get("selected_category")

        # ----- REFINEMENT STAGE (user selecting a sub‑option) -----
        if stage == STATE_REFINEMENT and selected_cat:
            matched = None
            for opt in INSURANCE_CATEGORIES.get(selected_cat, {}).get("options", []):
                if opt["label"].lower() in user_message.lower() or opt["id"] in user_message.lower():
                    matched = opt
                    break
            if matched:
                answer = await self._answer_for_policy_type(session_id, selected_cat, matched["label"], session.get("original_question", ""), history)
                self.sessions[session_id] = {"stage": STATE_DISCOVERY, "selected_category": None}
                return {
                    "message": answer,
                    "options": [],
                    "multi_select": False,
                    "next_question": "",
                    "intent": selected_cat,
                    "stage": "details"
                }, True
            else:
                # If the message looks like an unrelated question (not a sub-option selection),
                # reset state and process it as a fresh query instead of looping options again.
                q_lower = user_message.lower().strip()
                looks_like_question = (
                    len(user_message.split()) > 5
                    or "?" in user_message
                    or any(w in q_lower for w in ["will", "does", "is", "can", "how", "what", "why", "when", "do", "covered", "cover"])
                )
                if looks_like_question:
                    self.sessions[session_id] = {"stage": STATE_DISCOVERY, "selected_category": None}
                    # Fall through to intent classification below
                else:
                    options = self._get_sub_options(selected_cat)
                    return {
                        "message": f"Please select one of the options below for {INSURANCE_CATEGORIES[selected_cat]['label']} Insurance:",
                        "options": options,
                        "multi_select": False,
                        "next_question": FOLLOW_UP_QUESTIONS.get(selected_cat, "Which one interests you?"),
                        "intent": selected_cat,
                        "stage": STATE_REFINEMENT
                    }, False

        # ----- INTENT CLASSIFICATION -----
        intent = self._classify_intent(user_message)
        logger.info(f"Intent: {intent} | Message: {user_message}")

        # ----- KNOWLEDGE_BASE: user asking about capabilities / what I know / how I work -----
        if intent == "KNOWLEDGE_BASE":
            kb_msg = (
                "I'm loaded up with insurance knowledge across health, life, motor, travel, home "
                "and more! What would you like to explore?"
            )
            self._append_turn(session_id, user_message, kb_msg)
            return {
                "message": kb_msg,
                "options": [],
                "multi_select": False,
                "next_question": "",
                "intent": "knowledge_base",
                "stage": "details",
            }, True

        # ----- SMALL TALK: warm, hardcoded reply — no RAG, no LLM call -----
        if intent == "SMALL_TALK":
            import random
            _q = user_message.lower().strip().strip("!.,? ")
            if any(w in _q for w in ["bye", "goodbye", "see you", "later", "cya", "take care", "adios", "ttyl", "peace"]):
                small_talk_msg = random.choice([
                    "Take care! Feel free to come back anytime you have insurance questions. 😊",
                    "Bye! Whenever you need help with anything insurance-related, I'm right here.",
                    "See you! Don't hesitate to reach out if anything comes up.",
                ])
            elif any(w in _q for w in ["thank", "thanks", "ty", "thx", "cheers", "appreciate"]):
                small_talk_msg = random.choice([
                    "Happy to help! Let me know if there's anything else you want to know. 😊",
                    "Anytime! That's what I'm here for. Any other questions?",
                    "Glad I could help! Feel free to ask anything else.",
                ])
            elif any(w in _q for w in ["morning", "afternoon", "evening", "night", "gm", "gn"]):
                small_talk_msg = random.choice([
                    "Good to see you! How can I help you with insurance today?",
                    "Hey! Hope your day's going well. Got any insurance questions I can help with?",
                    "Hi! Great to chat. What can I help you with today?",
                ])
            elif any(w in _q for w in ["how are you", "how's it", "how are things", "you doing", "you good", "how do you do"]):
                small_talk_msg = random.choice([
                    "Doing great, thanks for asking! Ready to help you with any insurance questions. 😊",
                    "All good here! What can I help you with today?",
                    "I'm good! What insurance question can I sort out for you?",
                ])
            elif any(w in _q for w in ["start", "restart", "reset", "menu", "help", "back"]):
                small_talk_msg = "Sure! I can help you with insurance policies, coverage details, claims, and more. What would you like to know?"
            else:
                small_talk_msg = random.choice([
                    "Hey! 👋 I'm Layla, your insurance advisor from Nexsys IT Consulting. What can I help you with today?",
                    "Hi there! I'm Layla from Nexsys IT Consulting. Got an insurance question? I'm all ears. 😊",
                    "Hello! What insurance question can I help you with today?",
                    "Hey! Good to see you. What's on your mind?",
                ])
            return {
                "message": small_talk_msg,
                "options": [],
                "multi_select": False,
                "next_question": "",
                "intent": "small_talk",
                "stage": "details"
            }, True

        # ----- CASE C: INFORMATIONAL INTENT -----
        if intent == "INFORMATIONAL":
            doc_names = self._extract_document_names(user_message)
            session_history = self._build_history_string(session_id)
            try:
                answer, _, needs_human, is_off_topic = await self.multi_rag.ask(
                    user_message, session_history,
                    document_filter=doc_names if doc_names else None
                )
            except Exception as _rag_exc:
                logger.warning("multi_rag.ask() raised an unexpected error: %s", _rag_exc)
                err_msg = (
                    "I'm having trouble reaching my AI model server right now. "
                    "Please try again in a moment!"
                )
                self._append_turn(session_id, user_message, err_msg)
                return {
                    "message": err_msg,
                    "options": [],
                    "multi_select": False,
                    "next_question": "",
                    "intent": "informational",
                    "stage": "details",
                }, True

            if is_off_topic:
                off_topic_msg = (
                    "Ha, that's a bit outside my lane! 😄 I'm best with insurance "
                    "questions — policies, coverage, claims, premiums. Anything "
                    "insurance-related I can help you with?"
                )
                self._append_turn(session_id, user_message, off_topic_msg)
                return {
                    "message": off_topic_msg,
                    "options": [],
                    "multi_select": False,
                    "next_question": "",
                    "intent": "informational",
                    "stage": "details",
                }, True

            if needs_human:
                logger.warning(
                    "HANDOFF_TRIGGERED | session=%s | question=%s | intent=informational | "
                    "top_similarity <= 0.05 — no relevant context found",
                    session_id, user_message,
                )
                handoff_msg = (
                    "I'm not sure I can help with that one. Let me connect you "
                    "with one of our team members who can assist you further."
                )
                self._append_turn(session_id, user_message, handoff_msg)
                return {
                    "message": handoff_msg,
                    "options": [],
                    "multi_select": False,
                    "next_question": "",
                    "intent": "informational",
                    "stage": "details",
                    "needs_human": True,
                }, True

            self._append_turn(session_id, user_message, answer)
            return {
                "message": answer,
                "options": [],
                "multi_select": False,
                "next_question": "",
                "intent": "informational",
                "stage": "details"
            }, True
        # ----- CASE B: GENERAL INTENT (show all categories) -----
        if intent == "GENERAL":
            options = self._get_all_main_categories()
            return {
                "message": "Sure! Here are all the insurance categories I can help you with. Which one would you like to explore?",
                "options": options,
                "multi_select": False,
                "next_question": "Select a category to see available plans.",
                "intent": "general",
                "stage": STATE_DISCOVERY
            }, False

        # ----- CASE A & D: DISCOVERY or CATEGORY (show specific category options) -----
        category_id, category_label = self._extract_category_from_query(user_message)
        
        # If no category could be extracted, fallback to general menu
        if category_id == "unclear":
            options = self._get_all_main_categories()
            return {
                "message": "I'm not sure which insurance you're looking for. Here are all the options:",
                "options": options,
                "multi_select": False,
                "next_question": "Which category interests you?",
                "intent": "general",
                "stage": STATE_DISCOVERY
            }, False

        # Show sub‑options for that category
        options = self._get_sub_options(category_id)
        friendly_messages = {
            "motor": "Got it! 🚗 Let's find the right vehicle cover for you. Which one applies?",
            "health": "I understand health comes first. 💊 Here are the health plans we offer:",
            "life": "Securing your family's future is wise. 💼 Here are life insurance choices:",
            "travel": "Nice! ✈️ Here are the travel insurance options:",
            "home": "Protecting your home is important. 🏠 Here are the options:",
            "personal": "Accidents can happen. 🛡️ Here are personal accident/disability plans:"
        }
        message = friendly_messages.get(category_id, f"Here are the {category_label} insurance options available:")
        
        # Store session for next step (refinement)
        self.sessions[session_id] = {
            "stage": STATE_REFINEMENT,
            "selected_category": category_id,
            "original_question": user_message
        }
        return {
            "message": message,
            "options": options,
            "multi_select": False,
            "next_question": FOLLOW_UP_QUESTIONS.get(category_id, "Which specific plan are you interested in?"),
            "intent": category_id,
            "stage": STATE_DISCOVERY
        }, False

    # -------------------------------------------------------------------------
    # ANSWER FOR SELECTED SUB‑OPTION (using knowledge base)
    # -------------------------------------------------------------------------
    async def _answer_for_policy_type(self, session_id: str, category_id: str, sub_option_label: str, original_question: str, history: str = "") -> str:
        # Map sub‑option to policy_type metadata (for filtering)
        type_map = {
            "car": "motor", "bike": "motor", "commercial": "motor",
            "individual": "health", "family_floater": "health", "senior": "health", "critical": "health",
            "term": "life", "whole": "life", "ulip": "life", "endowment": "life",
            "international": "travel", "domestic": "travel", "student": "travel", "baggage": "baggage", "flight_delay": "flight_delay",
            "home": "home", "fire": "home", "property": "home",
            "accident": "personal", "disability": "personal"
        }
        policy_type = type_map.get(sub_option_label.lower().replace(" ", "_"), category_id)
        query = original_question or f"Tell me about {sub_option_label} insurance details, coverage, limits, exclusions."
        filter_meta = {"policy_type": policy_type}
        chunks = await asyncio.to_thread(
            self.vector_store.search,
            query, top_k=6, filter_metadata=filter_meta, use_hybrid=True, use_reranker=True,
        )
        if not chunks:
            return f"Thank you for your interest in {sub_option_label}. I couldn't find specific policy documents in the knowledge base. Please upload relevant insurance documents first."

        from rag import _build_structured_context
        from prompt_template import CONVERSATIONAL_RAG_PROMPT
        from router import get_insurance_llm
        context = _build_structured_context(chunks, max_chars=2000)
        prompt = CONVERSATIONAL_RAG_PROMPT.format(
            history=history,
            context=context,
            question=f"Provide a detailed overview of {sub_option_label} insurance policy, including what it covers, key benefits, exclusions, and conditions."
        )
        llm = get_insurance_llm(temperature=0.3)
        response = await asyncio.to_thread(llm.invoke, prompt)
        from multi_source_rag import _strip_model_preamble
        return _strip_model_preamble(response.content if hasattr(response, "content") else str(response))

    def restore_sessions(self, sessions: dict[str, dict]) -> None:
        """Restore persisted sessions on startup."""
        self.sessions = {str(k): dict(v) for k, v in sessions.items() if isinstance(v, dict)}

    def export_sessions(self) -> dict[str, dict]:
        """Export sessions for persistence."""
        return {str(k): dict(v) for k, v in self.sessions.items()}

    def reset_session(self, session_id: str) -> None:
        if session_id in self.sessions:
            del self.sessions[session_id]
