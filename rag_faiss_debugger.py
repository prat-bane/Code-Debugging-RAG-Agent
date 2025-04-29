"""
rag_faiss_hnsw_debugger.py
Fast, energy-lean FAISS RAG pipeline (HNSW-32).

✓ Uses the docstore + ID-map pattern recommended in LangChain docs
✓ fp32 vectors (HNSW expects float32)
✓ 4 CPU threads (adjust to your laptop)
✓ No CodeCarbon; add later if you want emissions tracking
"""

from __future__ import annotations
import os, pickle, pathlib, re
from typing import List, Dict

# --- minimal thread use -------------------------------------------------- #
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
torch.set_num_threads(4)

# --- LangChain / FAISS --------------------------------------------------- #
from langchain_huggingface import HuggingFaceEmbeddings  # pip install -U langchain-huggingface
from langchain_community.vectorstores import FAISS
from langchain_community.docstore import InMemoryDocstore
from langchain.schema import Document
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import faiss, numpy as np

# --- constants ----------------------------------------------------------- #
MODEL_NAME   = "thenlper/gte-small"       # fast 384-d encoder
CODE_DIR     = pathlib.Path("codebase")
INDEX_PKL    = pathlib.Path("faiss_hnsw.pkl")
LLM_URL      = "http://localhost:11434"
LLM_MODEL    = "llama3.2:1b"

# ------------------------------------------------------------------------- #
def _build_store(texts: List[str], metas: List[Dict]) -> FAISS:
    # 1) embeddings
    embed = HuggingFaceEmbeddings(model_name=MODEL_NAME,
                                   model_kwargs={"device": "cpu"})
    vecs  = np.asarray(embed.embed_documents(texts), dtype="float32")
    dim   = vecs.shape[1]

    # 2) HNSW index
    index = faiss.IndexHNSWFlat(dim, 32)        # M = 32 neighbours
    index.hnsw.efConstruction = 80
    index.add(vecs)
    index.hnsw.efSearch = 64

    # 3) docstore + ID map
    documents = [Document(page_content=t, metadata=m)
                 for t, m in zip(texts, metas)]
    docstore  = InMemoryDocstore({str(i): doc for i, doc in enumerate(documents)})
    id_map    = {i: str(i) for i in range(len(documents))}

    # 4) final vectorstore
    return FAISS(embedding_function=embed,
                 index=index,
                 docstore=docstore,
                 index_to_docstore_id=id_map)

def load_or_create_store() -> FAISS:
    if INDEX_PKL.exists():
        return pickle.loads(INDEX_PKL.read_bytes())

    texts, metas = [], []
    for f in CODE_DIR.glob("*.txt"):
        texts.append(f.read_text(encoding="utf-8"))
        metas.append({"source": f.name})

    store = _build_store(texts, metas)
    INDEX_PKL.write_bytes(pickle.dumps(store))
    print(f"[HNSW] Indexed {len(texts)} docs → {INDEX_PKL}")
    return store

STORE: FAISS = load_or_create_store()

# --- helpers ------------------------------------------------------------- #
def _retrieve(query: str, k: int = 5):
    return STORE.similarity_search(query, k=k)

def _error_info(log: str):
    typ = re.search(r"(?:Exception|Error): (.+?)(?:\n|$)", log)
    loc = re.search(r'File "(.*?)", line (\d+)', log)
    return {"error_type": typ.group(1).strip() if typ else "Unknown error",
            "location"  : f"{loc.group(1)}:{loc.group(2)}" if loc else "Unknown location"}

PROMPT = PromptTemplate(
    input_variables=["error_log", "error_type", "error_location", "code_files"],
    template="""You are a debugging assistant …

## Error Information
Error Log:
{error_log}

Error Type: {error_type}
Error Location: {error_location}

## Relevant Code
{code_files}

TASK:
1. Analyse the error and code.
2. Identify the root cause.
3. Provide a concise fix.
4. Explain why it works.

Your response:"""
)
LLM   = Ollama(base_url=LLM_URL, model=LLM_MODEL)
CHAIN = LLMChain(llm=LLM, prompt=PROMPT)

# --- public API ---------------------------------------------------------- #
def debug_error(error_log: str) -> str:
    docs  = _retrieve(error_log)
    files = "".join(f"\nFile {i+1}: {d.metadata['source']}\n```\n{d.page_content}\n```"
                    for i, d in enumerate(docs))
    info  = _error_info(error_log)
    return CHAIN.run(error_log=error_log,
                     error_type=info["error_type"],
                     error_location=info["location"],
                     code_files=files)

# --- CLI test ------------------------------------------------------------ #
if __name__ == "__main__":
    SAMPLE = """Traceback (most recent call last):
  File "main.py", line 42, in <module>
    run_app()
MemoryError: Simulated memory leak"""
    print("Vectors in index:", STORE.index.ntotal)
    print(debug_error(SAMPLE))
