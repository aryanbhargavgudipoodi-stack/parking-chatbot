"""
Stage 1 orchestrator: wires together intent routing, static RAG retrieval,
dynamic SQL lookups, reservation slot-filling, and the PII guardrail.

Collaborators (rag, reservation_agent, pii_filter, retriever) can all be
injected, which keeps this class fully unit-testable without any real
LLM/vector-DB/network calls (see tests/test_chatbot.py).
"""
import logging
import time

from src import sql_db
from src.guardrails.pii_filter import PIIFilter
from src.rag_chain import RagChain
from src.reservation import ReservationAgent
from src.vector_store import get_retriever, load_vector_store

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("parking_chatbot")

_BLOCKED_INPUT_ENTITIES = {"CREDIT_CARD", "US_SSN", "IBAN_CODE"}


class ParkingChatbot:
    def __init__(self, rag=None, reservation_agent=None, pii_filter=None, retriever=None, skip_sql_seed=False):
        if not skip_sql_seed:
            sql_db.seed_db()

        if rag is None and retriever is None:
            vectorstore = load_vector_store()
            retriever = get_retriever(vectorstore)

        self.rag = rag or RagChain(retriever)
        self.pii_filter = pii_filter or PIIFilter()
        self.reservation_agent = reservation_agent or ReservationAgent()
        self.mode = "qa"  # "qa" | "reservation"
        self.history = []

    def handle_message(self, message: str) -> str:
        start = time.time()

        if self.mode != "reservation":
            flagged = [e for e in self.pii_filter.scan(message) if e["entity_type"] in _BLOCKED_INPUT_ENTITIES]
            if flagged:
                response = (
                    "For your safety, please don't share financial or government ID "
                    "information in this chat. I only need your name, car number, "
                    "and reservation period to book a spot."
                )
                self._finish_turn(message, response, start)
                return response

        if self.mode == "reservation":
            response = self._continue_reservation(message)
        else:
            intent = self.rag.classify_intent(message)
            if intent == "static_info":
                response = self.rag.answer_static(message)["answer"]
            elif intent == "dynamic_info":
                response = self.rag.answer_dynamic(message)["answer"]
            elif intent == "reservation":
                self.mode = "reservation"
                response = self._continue_reservation(message)
            else:
                response = (
                    "I can help with parking info, prices, hours, availability, "
                    "location, or booking a reservation. How can I help?"
                )

            response, leaked = self.pii_filter.redact(response)
            if leaked:
                logger.warning("Redacted %d PII entity(ies) from outbound response", len(leaked))

        self._finish_turn(message, response, start)
        return response

    def _continue_reservation(self, message: str) -> str:
        self.reservation_agent.update_from_message(message)
        next_question = self.reservation_agent.next_question()
        if next_question:
            return next_question

        summary = self.reservation_agent.summary()
        self.mode = "qa"
        self.reservation_agent.reset()
        return (
            "Thanks! Here's what I have:\n"
            f"{summary}\n\n"
            "Your request has been queued for administrator approval "
            "(hand-off implemented in Stage 2)."
        )

    def _finish_turn(self, user_msg: str, bot_msg: str, start_time: float):
        latency = time.time() - start_time
        safe_user, _ = self.pii_filter.redact(user_msg)
        logger.info("USER: %s", safe_user)
        logger.info("BOT: %s", bot_msg)
        logger.info("Latency: %.3fs", latency)
        self.history.append({"user": user_msg, "bot": bot_msg, "latency_sec": latency})