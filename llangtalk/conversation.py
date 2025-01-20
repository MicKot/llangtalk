class Conversation:
    def __init__(self):
        self._prompts = []
        self._messages = []
        self._conversation_turns = []  # list of tuples (prompt, message)

    def add_prompt(self, prompt):
        self._prompts.append(prompt)

    def add_message(self, message):
        self._messages.append(message)
