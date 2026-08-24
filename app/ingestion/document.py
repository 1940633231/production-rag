from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class Document:

    document_id: str

    content: str

    metadata: Dict = field(default_factory=dict)

    @property
    def title(self) -> Optional[str]:
        return self.metadata.get("title")

    @property
    def source(self) -> Optional[str]:
        return self.metadata.get("source")

    @property
    def url(self) -> Optional[str]:
        return self.metadata.get("url")
