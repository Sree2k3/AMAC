from pydantic import BaseModel, Field
from typing import List, Optional

class Source(BaseModel):
    file: str
    startLine: int
    endLine: int

class AskResponse(BaseModel):
    answer: str
    sources: Optional[List[Source]] = []
    error: Optional[str] = None

