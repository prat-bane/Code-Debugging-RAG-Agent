from fastapi import FastAPI, UploadFile, File, HTTPException
from rag_debugger import debug_error, ensure_index   # import once, no heavy work
import uvicorn, asyncio

app = FastAPI(
    title="RAG Debug-API",
    description="POST a .txt log file and receive an automatic debugging solution.",
    version="0.1.0",
)

@app.on_event("startup")
def _warm_up():
    # Make sure the Marqo index exists; skip re-ingest if already there
    ensure_index("codebase")
    # Optionally: ping Ollama so the first request is not cold

@app.post("/debug", summary="Upload log file (.txt)")
async def debug(file: UploadFile = File(...)):
    if file.content_type not in {"text/plain", "application/octet-stream"}:
        raise HTTPException(415, "Only text files are accepted")

    log_text = (await file.read()).decode("utf-8", errors="ignore")
    # FastAPI default workers are threads → blocking call is okay
    solution = await asyncio.to_thread(debug_error, log_text)
    return {"solution": solution}

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
