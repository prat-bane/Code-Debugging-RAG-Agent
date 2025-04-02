from langchain_community.vectorstores import Marqo
import re
import os
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

def retrieve_relevant_code(marqo_client, error_log, top_k=5):
    """
    Given an error_log, use the Marqo vector store to find 'top_k' most relevant code documents.
    """

    # 1. Create a Marqo-based retriever
    retriever = Marqo(
        client=marqo_client,
        index_name="code_files",
    ).as_retriever(search_kwargs={"limit": top_k})

    # 2. Retrieve relevant documents based on the error log
    relevant_docs = retriever.get_relevant_documents(error_log)

    return relevant_docs



def extract_error_info(error_log):
    """
    Extract high-level error info such as error type and location (filename + line).
    """

    # 1. Error type extraction
    error_type_match = re.search(r'(?:Exception|Error): (.+?)(?:\n|$)', error_log)
    if error_type_match:
        error_type = error_type_match.group(1).strip()
    else:
        error_type = "Unknown error"

    # 2. File name + line extraction
    file_match = re.search(r'File "(.*?)", line (\d+)', error_log)
    if file_match:
        file_info = f"{file_match.group(1)}:{file_match.group(2)}"
    else:
        file_info = "Unknown location"

    return {
        "error_type": error_type,
        "location": file_info
    }




def generate_solution(error_log, code_files):
    # 1. Extract core error info
    error_info = extract_error_info(error_log)

    # 2. Configure Ollama LLM
    llm = Ollama(base_url="http://localhost:11434" ,model="llama3.2:1b")

    # 3. Create a structured prompt template
    prompt_template = PromptTemplate(
        input_variables=["error_log", "error_type", "error_location", "code_files"],
        template="""
You are a debugging assistant with expertise in fixing code issues.

## Error Information
Error Log:
{error_log}

Error Type: {error_type}
Error Location: {error_location}

## Relevant Code
{code_files}

TASK:
1. Analyze the error and the relevant code snippets.
2. Identify the root cause.
3. Provide a concise fix or recommendation.
4. Explain why this fix resolves the issue.

Your response:
"""
    )

    # 4. Format retrieved code for the prompt
    formatted_code = ""
    for i, doc in enumerate(code_files, start=1):
        source = doc.metadata.get("source", "Unknown Source")
        formatted_code += f"\nFile {i}: {source}\n```\n{doc.page_content}\n```"

    # 5. Build a Langchain chain
    chain = LLMChain(llm=llm, prompt=prompt_template)

    # 6. Run the chain and return the response
    response = chain.run(
        error_log=error_log,
        error_type=error_info["error_type"],
        error_location=error_info["location"],
        code_files=formatted_code
    )
    return response

def index_code_files(mq, index_name, directory):
    docs = []
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            with open(os.path.join(directory, filename), 'r') as f:
                content = f.read()
            docs.append({
                "source": filename,
                "text": content
            })
    # Use Marqo's API to add documents
    mq.index(index_name).add_documents(docs)
    print(f"Indexed {len(docs)} documents from {directory}")

def debug_error(marqo_client, error_log):
    # 1. Retrieve relevant code files from Marqo
    relevant_files = retrieve_relevant_code(marqo_client, error_log, top_k=5)

    # 2. Generate the debugging solution
    solution = generate_solution(error_log, relevant_files)
    return solution

if __name__ == "__main__":
    import marqo

    # A. Create or connect to Marqo client
    mq = marqo.Client("http://localhost:8882")
    INDEX_NAME = "code_files"

# Optionally delete the index if it exists (for development purposes)
    try:
        mq.delete_index(INDEX_NAME)
    except Exception as e:
        print("Index may not exist yet, creating a new one.")

    index_settings = {"model": "hf/e5-base-v2", "type": "structured", "allFields": [{"name": "text", "type": "text"},
                                                                                {"name":"source","type":"text"}],
    "tensorFields": ["text"],"normalizeEmbeddings": True,}

# Create the movie index
    mq.create_index(
        index_name=INDEX_NAME,
        # The exact keys can vary by Marqo version, but typically looks like:
        settings_dict=index_settings
    )
    index_code_files(mq, INDEX_NAME, "codebase")
    # B. Example error log
    sample_error_log = """
Traceback (most recent call last):
  File "main.py", line 42, in <module>
    run_app()
MemoryError: Simulated memory leak
    """

    # C. Use the debugging workflow
    solution = debug_error(mq, sample_error_log)
    print("=== DEBUG SOLUTION ===")
    print(solution)


