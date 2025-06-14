from dataclasses import dataclass
from datetime import datetime

@dataclass
class DialogInfo:
    dialog_id: int
    role: str
    text: str
    date: datetime = datetime.now()
    is_used: bool = False
    hint_type: str | None = None
    confidence: float | None = None
    source: str | None = None
    source_name: str | None = None

    def to_dict(self):
        return {
            "dialog_id": self.dialog_id,
            "role": self.role,
            "text": self.text,
            "date": self.date.isoformat(),
            "is_used": self.is_used,
            "hint_type": self.hint_type,
            "confidence": self.confidence,
            "source": self.source,
            "source_name": self.source_name,
        }