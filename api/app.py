from fastapi import FastAPI
from rag.chat import _get_response
from pydantic import BaseModel

app = FastAPI()

class QuestionRequest(BaseModel):
    question: str
    max_tokens: int = 1024

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    try: 
        answer, results = _get_response(request.question, top_k=5, max_tokens=request.max_tokens)
        return {
            "answer": answer,
            "sources": results
        }
    except Exception as e:
        return {
            "error": str(e)
        }