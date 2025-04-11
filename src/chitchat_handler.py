import re

class ChitChatHandler:
    def __init__(self):
        self.keywords = [
            r"^hi$",
            r"^hello$",
            r"^how are you$",
            r"^good morning$",
            r"^good evening$",
            r"^what's up$",
            r"^how's it going$",
            r"^thank you$",
            r"^thanks$",
            r"^bye$",
            r"^see you$"
        ]


    def is_chitchat(self, query: str) -> bool:
        q = normalize_query(query)
        return any(re.fullmatch(pattern, q) for pattern in self.keywords)
    

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

def normalize_query(self, q: str) -> str:
# Remove punctuation and extra spaces
    return re.sub(r'[^\w\s]', '', q.strip().lower())