from __future__ import annotations

from pathlib import Path


class DocumentLoader:
    def load(self, path: str) -> list[str]:
        file_path = Path(path)
        if not file_path.exists():
            return []
        return [file_path.read_text(encoding="utf-8")]
