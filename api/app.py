import dataclasses

from fastapi import FastAPI
from pydantic import BaseModel

from rag.chat import _get_response, _try_service_retriever
from rag.retriever import Retriever

app = FastAPI()

# Initialize once at startup, reuse across requests
_protocol_retriever = Retriever(icd_field="icd_codes")
_service_retriever = _try_service_retriever()


class QuestionRequest(BaseModel):
    question: str
    role: str = "patient"
    context: str = ""
    top_k: int = 5
    max_tokens: int = 1024
    category: str | None = None


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/ask")
async def ask_question(request: QuestionRequest):
    try:
        protocol_filters = {"category": request.category} if request.category else None

        answer, protocol_results, service_results = _get_response(
            raw_query=request.question,
            role=request.role,
            protocol_retriever=_protocol_retriever,
            service_retriever=_service_retriever,
            protocol_filters=protocol_filters,
            top_k=request.top_k,
            max_tokens=request.max_tokens,
            context_patient=request.context,
        )

        return {
            "answer": answer,
            "protocol_sources": [dataclasses.asdict(r) for r in protocol_results],
            "service_sources": [dataclasses.asdict(r) for r in service_results],
        }
    except Exception as e:
        return {"error": str(e)}