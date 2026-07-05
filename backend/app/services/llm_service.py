from __future__ import annotations

import json
import logging
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import httpx

from backend.app.core.config import get_settings

try:
    import nls
except ImportError:
    nls = None


settings = get_settings()
logger = logging.getLogger(__name__)


class DashScopeLLMService:
    def __init__(self) -> None:
        self.api_key = settings.qwen_api_key
        self.base_url = settings.qwen_base_url.rstrip("/")
        self.model = settings.qwen_chat_model
        self.audio_model = settings.qwen_audio_model
        self.asr_provider = settings.asr_provider.strip().lower()
        self.aliyun_nls_app_key = settings.aliyun_nls_app_key
        self.aliyun_nls_token = settings.aliyun_nls_token
        self.aliyun_nls_url = settings.aliyun_nls_url
        self.aliyun_access_key_id = settings.aliyun_access_key_id
        self.aliyun_access_key_secret = settings.aliyun_access_key_secret
        self.aliyun_nls_region = settings.aliyun_nls_region

    def available(self) -> bool:
        return bool(self.api_key)

    def chat(self, messages: list[dict[str, str]], temperature: float | None = None) -> str:
        if not self.available():
            return ""

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else settings.llm_temperature,
            "max_tokens": settings.llm_max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        try:
            with httpx.Client(timeout=15.0) as client:
                resp = client.post(f"{self.base_url}/chat/completions", headers=headers, json=payload)
                resp.raise_for_status()
                data: dict[str, Any] = resp.json()
        except (httpx.TimeoutException, httpx.HTTPError, ValueError):
            return ""

        choices = data.get("choices", [])
        if not choices:
            return ""
        message = choices[0].get("message", {})
        return message.get("content", "") or ""

    def extract_json(self, messages: list[dict[str, str]], temperature: float | None = 0.0) -> dict[str, Any]:
        raw = self.chat(messages, temperature=temperature)
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(raw[start : end + 1])
                except json.JSONDecodeError:
                    return {}
            return {}

    def transcribe_audio(self, audio_path: str, prompt: str | None = None) -> dict[str, Any]:
        path = Path(audio_path)
        if not path.exists():
            logger.warning("ASR input file missing: %s", audio_path)
            return {"text": "", "error": "音频文件不存在。"}

        if self.asr_provider == "aliyun_nls":
            return self._transcribe_audio_with_aliyun_nls(path)

        logger.warning("ASR provider unsupported: provider=%s", self.asr_provider)
        return {"text": "", "error": f"不支持的语音识别提供方：{self.asr_provider}"}

    def _transcribe_audio_with_aliyun_nls(self, path: Path) -> dict[str, Any]:
        if nls is None:
            logger.warning("Aliyun NLS SDK unavailable in current environment")
            return {
                "text": "",
                "error": "当前环境未安装 `nls` Python SDK，无法启用阿里云实时语音识别。请先按官方文档在 SDK 根目录执行 `python -m pip install .`，并确认代码中可以正常 `import nls`。",
            }

        if not self.aliyun_nls_app_key:
            logger.warning("Aliyun NLS unavailable: missing ALIYUN_NLS_APP_KEY")
            return {"text": "", "error": "当前未配置阿里云语音 AppKey，无法进行语音识别。"}

        if not self.aliyun_nls_token:
            logger.warning("Aliyun NLS unavailable: missing ALIYUN_NLS_TOKEN")
            return {
                "text": "",
                "error": "当前未配置阿里云语音 Token。根据官方实时语音识别 SDK 文档，服务端需要提供有效 Token，而不是仅有 AppKey。",
            }

        audio_bytes, conversion_error = self._load_audio_for_nls(path)
        if conversion_error:
            return {"text": "", "error": conversion_error}
        if not audio_bytes:
            return {
                "text": "",
                "error": "当前录音文件为空，或暂不支持该音频格式。阿里云实时语音识别示例推荐使用 16kHz 单声道 PCM 数据流。",
            }

        result_holder: dict[str, Any] = {
            "started": False,
            "completed": False,
            "error": "",
            "final_text": "",
            "sentence_texts": [],
            "raw_messages": [],
        }
        done_event = threading.Event()

        def on_start(message: str, *args: Any) -> None:
            logger.info("Aliyun NLS on_start: %s", message)
            result_holder["started"] = True
            result_holder["raw_messages"].append({"event": "start", "message": message})

        def on_sentence_begin(message: str, *args: Any) -> None:
            logger.info("Aliyun NLS on_sentence_begin: %s", message)
            result_holder["raw_messages"].append({"event": "sentence_begin", "message": message})

        def on_result_changed(message: str, *args: Any) -> None:
            logger.info("Aliyun NLS on_result_changed: %s", message)
            result_holder["raw_messages"].append({"event": "result_changed", "message": message})

        def on_sentence_end(message: str, *args: Any) -> None:
            logger.info("Aliyun NLS on_sentence_end: %s", message)
            result_holder["raw_messages"].append({"event": "sentence_end", "message": message})
            text = self._extract_aliyun_sentence_text(message)
            if text:
                result_holder["sentence_texts"].append(text)

        def on_completed(message: str, *args: Any) -> None:
            logger.info("Aliyun NLS on_completed: %s", message)
            result_holder["completed"] = True
            result_holder["raw_messages"].append({"event": "completed", "message": message})
            final_text = self._extract_aliyun_sentence_text(message)
            if final_text:
                result_holder["final_text"] = final_text
            done_event.set()

        def on_error(message: str, *args: Any) -> None:
            logger.error("Aliyun NLS on_error: %s", message)
            result_holder["error"] = self._extract_aliyun_error(message)
            result_holder["raw_messages"].append({"event": "error", "message": message})
            done_event.set()

        def on_close(*args: Any) -> None:
            logger.info("Aliyun NLS on_close: args=%s", args)

        logger.info(
            "Aliyun NLS realtime request started: file=%s size=%s url=%s app_key=%s",
            path.name,
            path.stat().st_size,
            self.aliyun_nls_url,
            self.aliyun_nls_app_key,
        )

        recognizer = None
        try:
            recognizer = nls.NlsSpeechTranscriber(
                url=self.aliyun_nls_url,
                token=self.aliyun_nls_token,
                appkey=self.aliyun_nls_app_key,
                on_sentence_begin=on_sentence_begin,
                on_sentence_end=on_sentence_end,
                on_start=on_start,
                on_result_changed=on_result_changed,
                on_completed=on_completed,
                on_error=on_error,
                on_close=on_close,
                callback_args=[path.name],
            )

            started = recognizer.start(
                aformat="pcm",
                sample_rate=16000,
                ch=1,
                enable_intermediate_result=True,
                enable_punctuation_prediction=True,
                enable_inverse_text_normalization=True,
                timeout=10,
            )
            logger.info("Aliyun NLS start returned: %s", started)
            if started is False and result_holder["started"]:
                logger.info("Aliyun NLS start returned False but on_start callback already succeeded: file=%s", path.name)
            if not result_holder["started"]:
                done_event.wait(timeout=3)

            if result_holder["error"]:
                return {"text": "", "error": result_holder["error"]}

            if not result_holder["started"]:
                logger.warning("Aliyun NLS start callback not received: file=%s start_return=%s", path.name, started)
                return {"text": "", "error": "阿里云语音识别未成功建立会话，请检查 Token、AppKey 和网关配置。"}

            send_failed = False
            for chunk in self._iter_audio_chunks(audio_bytes, 640):
                sent = recognizer.send_audio(chunk)
                if sent is False:
                    logger.warning("Aliyun NLS send_audio returned False: file=%s chunk_size=%s", path.name, len(chunk))
                    send_failed = True
                    break
                time.sleep(0.01)

            if send_failed and not result_holder["error"]:
                return {"text": "", "error": "阿里云语音流发送失败，请检查 Token 是否过期，或当前 PCM 音频是否符合 16kHz 单声道要求。"}

            stopped = recognizer.stop(timeout=10)
            logger.info("Aliyun NLS stop returned: %s", stopped)
            done_event.wait(timeout=12)

            if result_holder["error"]:
                return {"text": "", "error": result_holder["error"]}

            final_text = result_holder["final_text"] or "".join(result_holder["sentence_texts"]).strip()
            if not final_text:
                logger.warning("Aliyun NLS returned empty transcription: raw=%s", result_holder["raw_messages"])
                return {"text": "", "error": "阿里云语音识别已完成，但没有返回可用文本。请确认上传的是 16kHz 单声道 PCM 音频。"}

            logger.info("Aliyun NLS realtime request succeeded: file=%s text_length=%s", path.name, len(final_text))
            return {"text": final_text, "error": ""}
        except Exception as exc:
            logger.exception("Aliyun NLS realtime unexpected error: %s", str(exc))
            return {"text": "", "error": f"阿里云实时语音识别调用失败：{str(exc)}"}
        finally:
            if recognizer is not None:
                try:
                    recognizer.shutdown()
                except Exception:
                    logger.exception("Aliyun NLS shutdown failed")

    def _load_audio_for_nls(self, path: Path) -> tuple[bytes, str]:
        suffix = path.suffix.lower()
        if suffix == ".pcm":
            try:
                return path.read_bytes(), ""
            except OSError:
                logger.exception("Failed to read PCM audio file for NLS: %s", path)
                return b"", f"读取 PCM 音频文件失败：{path.name}"

        ffmpeg_path = shutil.which("ffmpeg")
        if not ffmpeg_path:
            logger.warning("ffmpeg not found while converting audio: file=%s suffix=%s", path.name, suffix)
            return (
                b"",
                "当前上传的是非 PCM 音频，但系统未安装 ffmpeg，无法自动转换为阿里云需要的 16kHz 单声道 PCM。请先安装 ffmpeg，或直接上传 PCM 音频。",
            )

        command = [
            ffmpeg_path,
            "-y",
            "-i",
            str(path),
            "-f",
            "s16le",
            "-acodec",
            "pcm_s16le",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-",
        ]
        logger.info("Converting audio to PCM with ffmpeg: file=%s command=%s", path.name, command)

        try:
            result = subprocess.run(command, capture_output=True, check=False, timeout=60)
        except subprocess.TimeoutExpired:
            logger.exception("ffmpeg conversion timed out: file=%s", path.name)
            return b"", "音频格式转换超时，请缩短录音长度后重试。"
        except OSError:
            logger.exception("Failed to execute ffmpeg: file=%s", path.name)
            return b"", "调用 ffmpeg 失败，请确认 ffmpeg 已正确安装并可执行。"

        if result.returncode != 0:
            stderr_text = result.stderr.decode("utf-8", errors="ignore").strip()
            logger.error("ffmpeg conversion failed: file=%s code=%s stderr=%s", path.name, result.returncode, stderr_text)
            detail = stderr_text[:300] if stderr_text else "无错误输出"
            return b"", f"音频格式转换失败：{detail}"

        pcm_bytes = result.stdout or b""
        if not pcm_bytes:
            logger.warning("ffmpeg conversion produced empty output: file=%s", path.name)
            return b"", "音频格式转换完成，但未生成有效的 PCM 数据。"

        logger.info("Audio conversion succeeded: file=%s pcm_bytes=%s", path.name, len(pcm_bytes))
        return pcm_bytes, ""

    def _iter_audio_chunks(self, data: bytes, chunk_size: int):
        for index in range(0, len(data), chunk_size):
            chunk = data[index : index + chunk_size]
            if chunk:
                yield chunk

    def _extract_aliyun_sentence_text(self, message: str) -> str:
        try:
            payload = json.loads(message)
        except json.JSONDecodeError:
            return ""

        if not isinstance(payload, dict):
            return ""

        payload_data = payload.get("payload") or {}
        if not isinstance(payload_data, dict):
            return ""

        return str(payload_data.get("result") or payload_data.get("text") or "").strip()

    def _extract_aliyun_error(self, message: str) -> str:
        try:
            payload = json.loads(message)
        except json.JSONDecodeError:
            return f"阿里云语音识别报错：{message}"

        if not isinstance(payload, dict):
            return f"阿里云语音识别报错：{message}"

        header = payload.get("header") or {}
        payload_data = payload.get("payload") or {}
        code = header.get("status") or header.get("code") or payload_data.get("status") or "unknown"
        detail = payload_data.get("message") or payload_data.get("error_message") or payload.get("message") or message
        return f"阿里云语音识别报错：{code} - {detail}"
