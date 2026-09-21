"""
request.py

What a client sends to the server (a subset of the OpenAI Responses API). The engine turns it into a `Sequence`.
"""
from pydantic import BaseModel


class ResponsesRequest(BaseModel):
    input: str | list[dict]
    max_output_tokens: int = 64
    temperature: float = 0.0

    def prompt(self) -> str:
        """Raw prompt text: `input` as-is, or the text of its message items concatenated."""
        if isinstance(self.input, str):
            return self.input
        texts = []
        for item in self.input:
            content = item.get("content", "")
            if isinstance(content, str):
                texts.append(content)
            else:
                texts.extend(part["text"] for part in content if "text" in part)
        return "".join(texts)
