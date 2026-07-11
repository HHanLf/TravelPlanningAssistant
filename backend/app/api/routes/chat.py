from __future__ import annotations

from pathlib import Path
import uuid

from fastapi import APIRouter, Depends, File, Form, UploadFile

from app.dependencies import get_travel_agent
from app.schemas.chat import ChatRequest, ChatResponse

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest, agent=Depends(get_travel_agent)) -> ChatResponse:
    data = await agent.handle(
        session_id=request.session_id,
        message=request.message,
        image_path=request.image_path,
        audio_path=request.audio_path,
        multimodal=request.multimodal,
    )
    return ChatResponse(**data)


@router.post("/multimodal", response_model=ChatResponse)
async def multimodal_chat(
    session_id: str = Form("default"),
    message: str = Form(""),
    image: UploadFile | None = File(default=None),
    audio: UploadFile | None = File(default=None),
    agent=Depends(get_travel_agent),
) -> ChatResponse:
    upload_dir = Path("data/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)

    image_path = None
    audio_path = None
    if image is not None:
        image_target = upload_dir / f"{uuid.uuid4().hex}_{Path(image.filename or 'image').name}"
        image_target.write_bytes(await image.read())
        image_path = str(image_target)
    if audio is not None:
        audio_target = upload_dir / f"{uuid.uuid4().hex}_{Path(audio.filename or 'audio').name}"
        audio_target.write_bytes(await audio.read())
        audio_path = str(audio_target)

    data = await agent.handle(
        session_id=session_id,
        message=message,
        image_path=image_path,
        audio_path=audio_path,
        multimodal=bool(image_path or audio_path),
    )
    return ChatResponse(**data)
