# rag_debugger.py
from langchain_community.vectorstores import Marqo
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import re, os
import marqo                               # direct SDK import

# ---------- Marqo helpers ---------- #

INDEX_NAME = "code_files"
MQ = marqo.Client("http://localhost:8882")

def ensure_index(directory: str = "codebase"):
    """Create the index and (re)ingest .txt documents if it doesn't exist."""
    if INDEX_NAME not in MQ.get_indexes()['results']:
        settings = {
            "model": "hf/e5-base-v2",
            "type": "structured",
            "allFields": [
                {"name": "text", "type": "text"},
                {"name": "source", "type": "text"}
            ],
            "tensorFields": ["text"],
            "normalizeEmbeddings": True,
        }
        MQ.create_index(index_name=INDEX_NAME, settings_dict=settings)

        # ingest
        docs = []
        for fname in os.listdir(directory):
            if fname.endswith(".txt"):
                with open(os.path.join(directory, fname), encoding="utf-8") as f:
                    docs.append({"source": fname, "text": f.read()})
        MQ.index(INDEX_NAME).add_documents(docs)
        print(f"Indexed {len(docs)} docs into {INDEX_NAME}")

# ---------- RAG pipeline ---------- #

def _retrieve_relevant_code(error_log: str, top_k: int = 5):
    retriever = Marqo(client=MQ, index_name=INDEX_NAME).as_retriever(
        search_kwargs={"limit": top_k}
    )
    return retriever.get_relevant_documents(error_log)

def _extract_error_info(error_log: str):
    typ = re.search(r"(?:Exception|Error): (.+?)(?:\n|$)", error_log)
    filepos = re.search(r'File "(.*?)", line (\d+)', error_log)
    return {
        "error_type": typ.group(1).strip() if typ else "Unknown error",
        "location": f"{filepos.group(1)}:{filepos.group(2)}" if filepos else "Unknown location",
    }

def _generate_solution(error_log: str, docs):
    llm = Ollama(base_url="http://localhost:11434", model="llama3.2:1b")

    prompt = PromptTemplate(
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

    formatted_code = ""
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source", "Unknown")
        formatted_code += f"\nFile {i}: {src}\n```\n{d.page_content}\n```"

    chain = LLMChain(llm=llm, prompt=prompt)
    info = _extract_error_info(error_log)
    return chain.run(
        error_log=error_log,
        error_type=info["error_type"],
        error_location=info["location"],
        code_files=formatted_code,
    )

def debug_error(error_log: str) -> str:
    """Public entry point: returns the LLM-generated solution string."""
    docs = _retrieve_relevant_code(error_log)
    return _generate_solution(error_log, docs)
