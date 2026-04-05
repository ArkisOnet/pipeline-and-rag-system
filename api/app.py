import dataclasses

from fastapi import FastAPI
from pydantic import BaseModel

from rag.chat import _get_response

app = FastAPI()


class QuestionRequest(BaseModel):
    question: str
    role: str = "patient"
    top_k: int = 5
    max_tokens: int = 1024


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/ask")
async def ask_question(request: QuestionRequest):
    try:
        answer, results = _get_response(
            request.question,
            role=request.role,
            top_k=request.top_k,
            max_tokens=request.max_tokens,
        )
        # SearchResult is a dataclass; convert to plain dicts for JSON serialization
        sources = [dataclasses.asdict(r) for r in results]
        return {"answer": answer, "sources": sources}
    except Exception as e:
        return {"error": str(e)}
