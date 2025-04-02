import marqo
import os

# Connect to the local Marqo server
mq = marqo.Client(url="http://localhost:8882")

INDEX_NAME = "code_files"

# Optionally delete the index if it exists (for development purposes)
try:
    mq.delete_index(INDEX_NAME)
except Exception as e:
    print("Index may not exist yet, creating a new one.")

index_settings = {"model": "hf/e5-base-v2", "type": "structured", "allFields": [{"name": "page_content", "type": "text"},
                                                                                {"name":"source","type":"text"}],
        "tensorFields": ["page_content"],"normalizeEmbeddings": True,}

# Create the movie index
mq.create_index(
    index_name=INDEX_NAME,
    # The exact keys can vary by Marqo version, but typically looks like:
    settings_dict=index_settings
)


def index_code_files(mq, index_name, directory):
    docs = []
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            with open(os.path.join(directory, filename), 'r') as f:
                content = f.read()
            docs.append({
                "source": filename,
                "page_content": content
            })
    # Use Marqo's API to add documents
    mq.index(index_name).add_documents(docs)
    print(f"Indexed {len(docs)} documents from {directory}")

# Example usage:
index_code_files(mq, INDEX_NAME, "codebase")

query_text = "Simulated memory leak"

# Perform a semantic (TENSOR) search
search_results = mq.index(INDEX_NAME).search(
    q=query_text,
    search_method="TENSOR",  # "TENSOR" for semantic, "LEXICAL" for keyword, or "HYBRID" for both
    limit=3                  # Number of search results to return
)

print(search_results)
