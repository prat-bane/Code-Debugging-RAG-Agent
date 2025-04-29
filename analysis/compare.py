import time
import numpy as np
import pandas as pd
import psutil
import os
import threading
import marqo
import faiss
from sentence_transformers import SentenceTransformer

# Reduce document size
NUM_DOCS = 50
MODEL_NAME = 'all-MiniLM-L6-v2'

# Create necessary folders
os.makedirs("./results", exist_ok=True)

class BenchmarkTest:
   def __init__(self, name, num_docs=NUM_DOCS):
       self.name = name
       self.num_docs = num_docs
       self.docs, self.queries = self._generate_test_data()
       # Separate results for indexing and search
       self.indexing_results = {
           "system": [],
           "time": [],
           "memory_usage_mb": []
       }
       self.search_results = {
           "system": [],
           "time": [],
           "memory_usage_mb": []
       }
       self.process = psutil.Process(os.getpid())
         
   def _generate_test_data(self):
       docs = [f"Document {i} with content for testing" for i in range(self.num_docs)]
       print(f'Created {self.num_docs} documents')
       queries = ["search query one", "another search query"]
       return docs, queries
         
   def _get_memory_usage(self):
       return self.process.memory_info().rss / (1024 * 1024)  # Convert to MB
         
   def record_indexing_result(self, system, time_taken):
       self.indexing_results["system"].append(system)
       self.indexing_results["time"].append(time_taken)
       self.indexing_results["memory_usage_mb"].append(self._get_memory_usage())
   
   def record_search_result(self, system, time_taken):
       self.search_results["system"].append(system)
       self.search_results["time"].append(time_taken)
       self.search_results["memory_usage_mb"].append(self._get_memory_usage())
         
   def get_indexing_df(self):
       return pd.DataFrame(self.indexing_results)
       
   def get_search_df(self):
       return pd.DataFrame(self.search_results)

class FAISSBenchmark:
   def __init__(self, model_name='all-MiniLM-L6-v2', index_type='flat'):
       self.model = SentenceTransformer(model_name)
       self.index = None
       self.documents = []
       self.index_type = index_type
           
   def index_documents(self, docs):
       start_time = time.time()
       
       # Create document embeddings
       embeddings = self.model.encode(docs)
       self.documents = docs
       
       # Create FAISS index based on type
       dimension = embeddings.shape[1]
       
       if self.index_type == 'flat':
           self.index = faiss.IndexFlatL2(dimension)
       elif self.index_type == 'hnsw':
           # Create HNSW index with fewer neighbors
           self.index = faiss.IndexHNSWFlat(dimension, 16)  # Reduced from 32
       else:
           self.index = faiss.IndexFlatL2(dimension)
       
       self.index.add(np.asarray(embeddings, dtype=np.float32))
       
       end_time = time.time()
       
       return {
           "time": end_time - start_time,
       }
           
   def search(self, queries, k=5):  # Reduced k from 10
       start_time = time.time()
       
       results = []
       for query in queries:
           # Encode query
           query_vector = self.model.encode([query])
           
           # Search
           distances, indices = self.index.search(np.asarray(query_vector, dtype=np.float32), k)
           
           query_results = []
           for i, idx in enumerate(indices[0]):
               if idx != -1 and idx < len(self.documents):
                   query_results.append({
                       "document": self.documents[idx],
                       "distance": float(distances[0][i])
                   })
           results.append(query_results)
       
       end_time = time.time()
       
       return {
           "results": results,
           "time": end_time - start_time,
       }

class MarqoBenchmark:
   def __init__(self, index_name="marqo-benchmark"):
       try:
           self.client = marqo.Client(url="http://localhost:8882")
           self.index_name = index_name
           
           settings = {
                "treat_urls_and_pointers_as_images": False,
                "model": MODEL_NAME  # Use same model as FAISS
            }
           
           
           # Create index if it doesn't exist
           try:
               self.client.create_index(self.index_name)
           except:
               # Delete index if it exists
               try:
                   self.client.delete_index(self.index_name)
                   self.client.create_index(self.index_name)
               except:
                   pass
       except Exception as e:
           print(f"Failed to initialize Marqo: {e}")
           raise e
         
   def index_documents(self, docs):
       start_time = time.time()
       
       # Convert to documents format
       documents = [{"id": str(i), "text": doc} for i, doc in enumerate(docs)]
       
       # Add documents in smaller batches
       batch_size = 20
       for i in range(0, len(documents), batch_size):
           batch = documents[i:i+batch_size]
           self.client.index(self.index_name).add_documents(batch, tensor_fields=['text'])
       
       end_time = time.time()
       
       return {
           "time": end_time - start_time,
       }
         
   def search(self, queries, k=5):  # Reduced k from 10
       start_time = time.time()
       
       results = []
       for query in queries:
           # Search
           query_results = self.client.index(self.index_name).search(
               q=query,
               limit=k
           )
           results.append(query_results.get("hits", []))
       
       end_time = time.time()
       
       return {
           "results": results,
           "time": end_time - start_time,
       }

def run_benchmark(num_docs=NUM_DOCS):
   benchmark = BenchmarkTest("Vector Database Comparison", num_docs)
   
   # Test FAISS Flat
   print(f"Testing FAISS Flat with {num_docs} documents...")
   
   faiss_benchmark = FAISSBenchmark(index_type='flat')
   
   # Index documents
   index_results = faiss_benchmark.index_documents(benchmark.docs)
   benchmark.record_indexing_result(
       "FAISS Flat", 
       index_results["time"]
   )
   
   # Search
   search_results = faiss_benchmark.search(benchmark.queries)
   benchmark.record_search_result(
       "FAISS Flat", 
       search_results["time"]
   )
   
   # Test FAISS HNSW
   print(f"Testing FAISS HNSW with {num_docs} documents...")
   
   faiss_hnsw_benchmark = FAISSBenchmark(index_type='hnsw')
   
   # Index documents
   index_results = faiss_hnsw_benchmark.index_documents(benchmark.docs)
   benchmark.record_indexing_result(
       "FAISS HNSW", 
       index_results["time"]
   )
   
   # Search
   search_results = faiss_hnsw_benchmark.search(benchmark.queries)
   benchmark.record_search_result(
       "FAISS HNSW", 
       search_results["time"]
   )
   
   # Test Marqo
   try:
       print(f"Testing Marqo with {num_docs} documents...")
       
       marqo_benchmark = MarqoBenchmark()
       
       # Index documents
       index_results = marqo_benchmark.index_documents(benchmark.docs)
       benchmark.record_indexing_result(
           "Marqo", 
           index_results["time"]
       )
       
       # Search
       search_results = marqo_benchmark.search(benchmark.queries)
       benchmark.record_search_result(
           "Marqo", 
           search_results["time"]
       )
   except Exception as e:
       print(f"Marqo error: {e}")
       print("Make sure Marqo is running: docker run -p 8882:8882 marqoai/marqo:latest")
   
   # Return separate dataframes
   indexing_df = benchmark.get_indexing_df()
   search_df = benchmark.get_search_df()
   return indexing_df, search_df

if __name__ == "__main__":
   try:
       print("\nRunning benchmark")
       indexing_results, search_results = run_benchmark()
       
       # Print and save summary
       print("\nIndexing Results:")
       print(indexing_results)
       indexing_results.to_csv("./results/indexing_results.csv", index=False)
       
       print("\nSearch Results:")
       print(search_results)
       search_results.to_csv("./results/search_results.csv", index=False)
       
       # Create plots
       try:
           import matplotlib.pyplot as plt
           import seaborn as sns
           
           os.makedirs("./results/plots", exist_ok=True)
           
        #    # Indexing time comparison
        #    plt.figure(figsize=(10, 5))
        #    sns.barplot(x="system", y="time", data=indexing_results)
        #    plt.title("Indexing Time Comparison")
        #    plt.ylabel("Time (seconds)")
        #    plt.savefig("./results/plots/indexing_time.png")
        #    plt.close()
        
            # Indexing time comparison (without Marqo)
           plt.figure(figsize=(10, 5))
           faiss_indexing = indexing_results[indexing_results["system"] != "Marqo"]
           ax = sns.barplot(x="system", y="time", data=faiss_indexing)
           plt.title(f"FAISS Indexing Time Comparison ({MODEL_NAME})")
           plt.ylabel("Time (seconds)")
           for i in ax.containers:
               ax.bar_label(i)
           plt.savefig("./results/plots/faiss_indexing_time.png")
           plt.close()
           
           # Indexing memory comparison
           plt.figure(figsize=(10, 5))
           sns.barplot(x="system", y="memory_usage_mb", data=indexing_results)
           plt.title("Indexing Memory Usage Comparison")
           plt.ylabel("Memory (MB)")
           plt.savefig("./results/plots/indexing_memory.png")
           plt.close()
           
           # Search time comparison
           plt.figure(figsize=(10, 5))
           sns.barplot(x="system", y="time", data=search_results)
           plt.title("Search Time Comparison")
           plt.ylabel("Time (seconds)")
           plt.savefig("./results/plots/search_time.png")
           plt.close()
           
           # Search memory comparison
           plt.figure(figsize=(10, 5))
           sns.barplot(x="system", y="memory_usage_mb", data=search_results)
           plt.title("Search Memory Usage Comparison")
           plt.ylabel("Memory (MB)")
           plt.savefig("./results/plots/search_memory.png")
           plt.close()
           
           print("Created separate charts in ./results/plots/")
       except Exception as e:
           print(f"Error creating plots: {e}")
           
   except Exception as e:
       print(f"Error during benchmark: {e}")