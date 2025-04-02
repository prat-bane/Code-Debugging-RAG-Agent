# Code-Debugging-RAG-Agent

This repository demonstrates a **Retrieval-Augmented Generation (RAG)** workflow that uses [Marqo](https://github.com/marqo-ai/marqo) as the vector store for code documents and [Ollama](https://github.com/jmorganca/ollama) as the Large Language Model (LLM) generator. The pipeline leverages [Langchain Community](https://github.com/langchain4j/langchain-community) wrappers to seamlessly integrate both Marqo and Ollama.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Setup & Usage](#setup--usage)
  - [1. Run Marqo](#1-run-marqo)
  - [2. Index Code Files](#2-index-code-files)
  - [3. Configure Ollama](#3-configure-ollama)
  - [4. Provide Error Log](#4-provide-error-log)
  - [5. Run the Script](#5-run-the-script)
- [Key Components](#key-components)
- [How It Works](#how-it-works)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Overview

This project aims to **debug code errors** using a Retrieval-Augmented Generation approach. You supply an error log, and the system:

1. **Queries Marqo** for the top relevant code documents.  
2. **Builds a structured prompt** combining the error log, error details, and the retrieved code.  
3. **Sends the prompt to Ollama**, which generates a debugging solution or proposed fix.

---

## Features

- **Semantic Search with Marqo**: Documents (code files) are vectorized and stored in Marqo.  
- **Structured Prompting**: The error log and relevant snippets are combined into a single, context-rich prompt.  
- **Ollama LLM Integration**: The retrieved context is passed to Ollama, which generates the debugging solution.  
- **Regex-based Error Extraction**: Simple functions parse error types and locations from the log.  

---

## Project Structure

- **`codebase/`**: Directory containing text files or code files you want to index (e.g., `.txt`, `.py`).
- **`retriever.py`**: Main script tying together Marqo indexing, retrieval, and Ollama LLM invocation.
- **`requirements.txt`**: Python dependencies (Langchain Community, Marqo, etc.).
- **`README.md`**: This file (project documentation).

---

## Installation

1. **Clone This Repository**:

   ```bash
   git clone https://github.com/your-username/your-repo.git
   cd your-repo


