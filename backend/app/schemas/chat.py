from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    session_id: str = Field(default="default")
    message: str = Field(min_length=1)
    image_path: str | None = None
    audio_path: str | None = None
    multimodal: bool = False


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    intent: dict
    plan: dict
    tool_results: dict
    memory_context: dict
    reflection_result: dict
    retrieved_docs: list[dict] = Field(default_factory=list)
    profile: dict = Field(default_factory=dict)
    multimodal_summary: str | None = None
    audio_transcript: str | None = None
