from __future__ import annotations

from pathlib import Path
import uuid

from fastapi import APIRouter, File, Form, UploadFile

from backend.app.schemas.chat import ChatRequest, ChatResponse
from backend.app.services.agent_service import AgentService

router = APIRouter()
service = AgentService()


@router.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    return ChatResponse(
        **service.chat(
            payload.session_id,
            payload.message,
            image_path=payload.image_path,
            audio_path=payload.audio_path,
            multimodal=payload.multimodal,
        )
    )


@router.post("/chat/multimodal", response_model=ChatResponse)
async def chat_multimodal(
    session_id: str = Form(default="default"),
    message: str = Form(default=""),
    image: UploadFile | None = File(default=None),
    audio: UploadFile | None = File(default=None),
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

    return ChatResponse(
        **service.chat(
            session_id,
            message,
            image_path=image_path,
            audio_path=audio_path,
            multimodal=bool(image_path or audio_path),
        )
    )


@router.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "travel-planning-assistant"}
