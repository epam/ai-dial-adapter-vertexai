class ValidationError(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


class UserError(Exception):
    def __init__(self, message: str, usage: str):
        self.message = message
        self.usage = usage
        super().__init__(self.message)

    def to_message_for_chat_user(self) -> str:
        return f"{self.message}\n\n{self.usage}"
