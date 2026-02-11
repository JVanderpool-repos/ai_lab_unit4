
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

    sentences = [
        # Animals and pets
        "The canine barked loudly.",
        "The dog made a noise.",
        "Puppies need lots of attention and exercise.",
        "Cats enjoy sleeping in sunny spots.",
        "Birds sing beautifully in the morning.",
        "Fish swim gracefully in the aquarium.",
        # Science and physics
        "Quantum mechanics explains particle behavior.",
        "Atoms are made of protons, neutrons, and electrons.",
        "The electron spins rapidly.",
        # Food and cooking
        "Chocolate cake is delicious.",
        "I love making pasta for dinner.",
        "Fresh fruit is healthy and tasty.",
        # Sports and activities
        "Soccer is a popular sport worldwide.",
        "Swimming is great exercise for the body.",
        # Weather and nature
        "The ocean is very deep.",
        "Rain falls gently on the rooftop.",
        # Technology and programming
        "I love programming in Python.",
        "Artificial intelligence is changing the world."
    ]

    print(f"Storing {len(sentences)} sentences in the vector database...")

    # Prepare metadata for each sentence
    now = datetime.datetime.now().isoformat()
    metadatas = [
        {"created_at": now, "index": idx}
        for idx in range(len(sentences))
    ]

    # Create InMemoryVectorStore instance
    vector_store = InMemoryVectorStore(embeddings)

    # Add sentences to vector store with metadata
    vector_store.add_texts(sentences, metadatas=metadatas)

    print(f"✅ Successfully stored {len(sentences)} sentences\n")

    # Interactive semantic search loop
    print("=== Semantic Search ===\n")
    while True:
        user_query = input("Enter a search query (or 'quit' to exit): ")
        if user_query.strip().lower() in {"quit", "exit"}:
            print("\n👋 Goodbye!")
            break
        if not user_query.strip():
            continue
        search_sentences(vector_store, user_query)
        print()
    # ...existing code...

if __name__ == "__main__":
    main()