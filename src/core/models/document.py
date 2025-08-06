from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict
import uuid


@dataclass
class Document:
    """Lightweight container for text snippets and associated metadata.

    Attributes:
        text: The textual content of the document or chunk.
        metadata: Arbitrary key/value metadata for downstream processors.
        id: Stable unique identifier generated if not supplied.
    """

    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
