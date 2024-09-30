# Retrieval-Augmented Generation (RAG) Project

This project demonstrates a basic implementation of **Retrieval-Augmented Generation (RAG)** using FAISS for document retrieval and GPT-2 for text generation. The goal of RAG is to improve the accuracy and relevance of text generation by retrieving relevant documents from an external source and incorporating them into the generation process.

## Features

- **Document Retrieval**: Indexes documents using FAISS for fast similarity search.
- **Augmented Text Generation**: Uses GPT-2 to generate responses based on both the query and retrieved documents.
- **Simple and Extendable**: A basic example of RAG that can be extended to larger document stores and different generation models.

## Setup

### Prerequisites

- Python 3.7+
- pip
- A machine with a compatible **Radeon GPU** or **CPU** setup.

### Install Dependencies

1. Clone the repository:
    ```bash
    git clone https://github.com/AbheestaV/rag-bot.git
    cd rag-implementation
    ```

2. Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```