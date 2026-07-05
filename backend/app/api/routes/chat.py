from fastapi import APIRouter, Depends, File, Form, UploadFile

from backend.app.dependencies import get_travel_agent
from backend.app.schemas.chat import ChatRequest, ChatResponse

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest, agent=Depends(get_travel_agent)) -> ChatResponse:
    data = await agent.handle(session_id=request.session_id, message=request.message)
    return ChatResponse(**data)


@router.post("/multimodal", response_model=ChatResponse)
async def multimodal_chat(
    session_id: str = Form("default"),
    message: str = Form(""),
    audio: UploadFile | None = File(default=None),
    agent=Depends(get_travel_agent),
) -> ChatResponse:
    audio_transcript = f"[收到音频文件: {audio.filename}]" if audio else None
    prompt = message or "请结合上传内容生成旅行建议"
    data = await agent.handle(session_id=session_id, message=prompt, audio_transcript=audio_transcript)
    return ChatResponse(**data)
