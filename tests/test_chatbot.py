from src.chatbot import ParkingChatbot


class _FakeRag:
    def __init__(self, intent="static_info", answer="Static answer."):
        self.intent = intent
        self.answer = answer

    def classify_intent(self, _message):
        return self.intent

    def answer_static(self, _q):
        return {"answer": self.answer, "sources": ["doc1"]}

    def answer_dynamic(self, _q):
        return {"answer": self.answer, "sources": ["sql:parking_lots"]}


class _FakeReservationAgent:
    def __init__(self, done=False):
        self.updates = []
        self._done = done

    def update_from_message(self, message):
        self.updates.append(message)

    def next_question(self):
        return None if self._done else "Could you tell me your first name?"

    def summary(self):
        return "Name: Jane Doe\nCar number: ABC123"

    def reset(self):
        pass


class _FakePIIFilter:
    def scan(self, text):
        if "4111111111111111" in text:
            return [{"entity_type": "CREDIT_CARD", "start": 0, "end": 16, "score": 0.9}]
        return []

    def redact(self, text):
        return text, []


def _make_bot(**overrides):
    defaults = dict(
        rag=_FakeRag(), reservation_agent=_FakeReservationAgent(),
        pii_filter=_FakePIIFilter(), retriever=object(), skip_sql_seed=True,
    )
    defaults.update(overrides)
    return ParkingChatbot(**defaults)


def test_blocks_unsafe_input_outside_reservation_mode():
    bot = _make_bot()
    response = bot.handle_message("here is my card 4111111111111111")
    assert "safety" in response.lower()


def test_static_info_intent_routes_to_answer_static():
    bot = _make_bot(rag=_FakeRag(intent="static_info", answer="Downtown opens at 6am."))
    response = bot.handle_message("When does downtown garage open?")
    assert "6am" in response


def test_reservation_intent_switches_mode_and_asks_question():
    bot = _make_bot(rag=_FakeRag(intent="reservation"))
    response = bot.handle_message("I'd like to book a spot")
    assert bot.mode == "reservation"
    assert "name" in response.lower()


def test_reservation_completion_returns_to_qa_mode():
    bot = _make_bot(reservation_agent=_FakeReservationAgent(done=True))
    bot.mode = "reservation"
    response = bot.handle_message("Jane Doe ABC123 tomorrow 9-11am")
    assert bot.mode == "qa"
    assert "administrator" in response.lower()