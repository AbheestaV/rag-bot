import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModel

# Prepare a small document store
documents = [
    "Python is a language",
    "This is an AI Project",
    "Hugging Face is a company creating AI Technologies",
    "Transformers are deep learning models that handle data"
]

# Tokenizer and model for creating document embeddings
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')

# Function to generate embeddings for the documents
def get_embeddings(documents):
    inputs = tokenizer(documents, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings.cpu().numpy()

# Generate document embeddings
doc_embeddings = get_embeddings(documents)

# Set up FAISS index
index = faiss.IndexFlatL2(doc_embeddings.shape[1])
index.add(doc_embeddings)


# Retrieval function
def retrieve(query, top_k=2):
    query_embedding = get_embeddings([query])
    distances, indices = index.search(query_embedding, top_k)
    return [(documents[i], distances[0][i]) for i in indices[0]]

# GPT-2 for response generation
gpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt_model = GPT2LMHeadModel.from_pretrained("gpt2")

# Full RAG process
def rag(query):
    retrieved_docs = retrieve(query)
    augmented_query = query + " " + " ".join([doc for doc, _ in retrieved_docs])
    inputs = gpt_tokenizer.encode(augmented_query, return_tensors='pt')
    gpt_output = gpt_model.generate(inputs, max_length=100, num_return_sequences=1)
    response = gpt_tokenizer.decode(gpt_output[0], skip_special_tokens=True)
    return response

# Example query
query = "Tell me about deep learning models."
response = rag(query)
print(f"RAG Response: {response}")