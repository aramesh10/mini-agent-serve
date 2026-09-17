@dataclass(slots=True)
class Request:
    id: int
    prompt: str
    