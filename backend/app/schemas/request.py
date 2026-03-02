from pydantic import BaseModel, Field

class AskRequest(BaseModel):
    repo: str = Field(..., description="GitHub repo URL")
    question: str = Field(..., description="User's natural-language question")

