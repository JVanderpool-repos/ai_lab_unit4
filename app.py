
import os
import math
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
import datetime



# Load environment variables
load_dotenv()

def cosine_similarity(vector_a, vector_b):
    """
    Calculate cosine similarity between two vectors
    Cosine similarity = (A · B) / (||A|| * ||B||)
    """
    if len(vector_a) != len(vector_b):
        raise ValueError("Vectors must have the same dimensions")
    
    dot_product = sum(a * b for a, b in zip(vector_a, vector_b))
    norm_a = math.sqrt(sum(a * a for a in vector_a))
    norm_b = math.sqrt(sum(b * b for b in vector_b))
    
    return dot_product / (norm_a * norm_b)

def search_sentences(vector_store, query, k=3):
    """
    Search for similar sentences in the vector store.
    Returns top k results with similarity scores.
    """
    results = vector_store.similarity_search_with_score(query, k=k)
    print(f"\n🔍 Search Results for \"{query}\":\n")
    for idx, (doc, score) in enumerate(results, 1):
        print(f"{idx}. [Score: {score:.4f}] {doc.page_content}")
    return results


# Helper function to load a document and add to vector store
def load_document(vector_store, file_path):
    """
    Loads a document from file_path, creates a LangChain Document, adds it to the vector store, and returns the document ID.
    Handles errors and prints a success message.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        metadata = {
            'fileName': os.path.basename(file_path),
            'createdAt': datetime.datetime.now().isoformat()
        }
        document = Document(page_content=content, metadata=metadata)
        doc_ids = vector_store.add_documents([document])
        print(f"✅ Loaded '{metadata['fileName']}' ({len(content)} chars) into vector store.")
        return doc_ids[0] if doc_ids else None
    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
        return None
    except Exception as e:
        error_msg = str(e)
        if "maximum context length" in error_msg or "token" in error_msg:
            print(f"⚠️ This document is too large to embed as a single chunk.")
            print("Token limit exceeded. The embedding model can only process up to 8,191 tokens at once.")
            print("Solution: The document needs to be split into smaller chunks.")
        else:
            print(f"❌ Error loading document '{file_path}': {error_msg}")
        return None

def main():
    print("🤖 Python LangChain Agent Starting...\n")

    # Check for GitHub token
    if not os.getenv("GITHUB_TOKEN"):
        print("❌ Error: GITHUB_TOKEN not found in environment variables.")
        print("Please create a .env file with your GitHub token:")
        print("GITHUB_TOKEN=your-github-token-here")
        print("\nGet your token from: https://github.com/settings/tokens")
        print("Or use GitHub Models: https://github.com/marketplace/models")
        return


    # Create OpenAIEmbeddings instance for GitHub Models API
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        base_url="https://models.inference.ai.azure.com",
        api_key=os.getenv("GITHUB_TOKEN"),
        check_embedding_ctx_length=False
    )

    print("=== Vector Store Lab ===\n")

    # Create InMemoryVectorStore instance
    vector_store = InMemoryVectorStore(embeddings)

    # Load documents into the vector database
    print("=== Loading Documents into Vector Database ===\n")
    file_to_load = "HealthInsuranceBrochure.md"
    doc_id = load_document(vector_store, file_to_load)
    if doc_id:
        print(f"Document '{file_to_load}' loaded successfully with ID: {doc_id}\n")
    else:
        print(f"Failed to load document: {file_to_load}\n")

    # Load EmployeeHandbook.md as well
    file_to_load2 = "EmployeeHandbook.md"
    doc_id2 = load_document(vector_store, file_to_load2)
    if doc_id2:
        print(f"Document '{file_to_load2}' loaded successfully with ID: {doc_id2}\n")
    else:
        print(f"Failed to load document: {file_to_load2}\n")
    # ...existing code...

if __name__ == "__main__":
    main()