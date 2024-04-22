class ValidationError(Exception):
    message: str

    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


class UserError(Exception):
    display_message: str
    usage: str

    def __init__(self, display_message: str, usage: str):
        self.display_message = display_message
        self.usage = usage
        super().__init__(self.display_message)
