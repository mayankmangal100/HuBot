class ChitChatHandler:
    def __init__(self):
        self.keywords = [
            "hi", "hello", "how are you", "good morning", "good evening",
            "what's up", "how's it going", "thank you", "bye", "see you"
        ]

    def is_chitchat(self, query: str) -> bool:
        """Check if the query is likely a chit-chat."""
        q = query.lower()
        return any(kw in q for kw in self.keywords)

    def handle(self, query: str) -> str:
        """Return a canned response based on chit-chat type."""
        q = query.lower()

        if "how are you" in q:
            return "I'm just a bot, but I'm here to help!"
        elif "hi" in q or "hello" in q:
            return "Hi there! How can I assist you today?"
        elif "thank" in q:
            return "You're very welcome!"
        elif "bye" in q or "see you" in q:
            return "Goodbye! Have a great day!"
        elif "good morning" in q:
            return "Good morning! Hope you're having a great start to the day!"
        elif "good evening" in q:
            return "Good evening! How can I help you tonight?"
        elif "what's up" in q or "how's it going" in q:
            return "All good here! What can I help you with?"
        else:
            return "I'm here to answer your questions!"

