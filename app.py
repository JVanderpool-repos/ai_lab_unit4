
import os
import math
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langchain.agents import create_agent
import datetime

from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter, MarkdownHeaderTextSplitter
# Optionally, for other splitters:
# from langchain_text_splitters import TokenTextSplitter


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

def load_document_with_chunks(vector_store, file_path, chunks):
    """
    Loads chunks of a document into the vector store with metadata.
    Updates each chunk's metadata with file name, chunk index, and creation timestamp.
    
    Args:
        vector_store: The vector store to add documents to
        file_path: Path to the original document file
        chunks: List of LangChain Document objects (chunks)
    
    Returns:
        Total number of chunks successfully stored
    """
    try:
        file_name = os.path.basename(file_path)
        total_chunks = len(chunks)
        added_count = 0
        
        for idx, chunk in enumerate(chunks, 1):
            # Update metadata for each chunk
            chunk.metadata['fileName'] = f"{file_name} (Chunk {idx}/{total_chunks})"
            chunk.metadata['createdAt'] = datetime.datetime.now().isoformat()
            chunk.metadata['chunkIndex'] = idx
            
            # Add chunk to vector store
            doc_ids = vector_store.add_documents([chunk])
            if doc_ids:
                added_count += 1
                print(f"✅ Added chunk {idx}/{total_chunks} from '{file_name}'")
        
        print(f"\n✅ Successfully stored {added_count}/{total_chunks} chunks from '{file_name}'")
        return added_count
    except Exception as e:
        print(f"❌ Error loading document chunks from '{file_path}': {str(e)}")
        return 0

def load_with_fixed_size_chunking(vector_store, file_path):
    """
    Reads a document file and splits it into fixed-size chunks using CharacterTextSplitter.
    Creates Document objects from chunks and adds them to the vector store.
    Prints statistics about the chunking process.
    
    Args:
        vector_store: The vector store to add documents to
        file_path: Path to the document file to read and chunk
    
    Returns:
        Total number of chunks successfully stored
    """
    try:
        # Read the file
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Create CharacterTextSplitter with specified parameters
        splitter = CharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=0,
            separator=" "
        )
        
        # Create Document objects from the text
        chunks = splitter.create_documents([text])
        
        # Calculate statistics
        num_chunks = len(chunks)
        avg_chunk_size = len(text) / num_chunks if num_chunks > 0 else 0
        
        print(f"📊 Chunking Statistics for '{os.path.basename(file_path)}':")
        print(f"   - Total chunks created: {num_chunks}")
        print(f"   - Average chunk size: {avg_chunk_size:.1f} characters")
        print()
        
        # Pass chunks to load_document_with_chunks
        return load_document_with_chunks(vector_store, file_path, chunks)
    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
        return 0
    except Exception as e:
        print(f"❌ Error in load_with_fixed_size_chunking for '{file_path}': {str(e)}")
        return 0

def load_with_recursive_chunking(vector_store, file_path):
    """
    Reads a document file and splits it into chunks using RecursiveCharacterTextSplitter.
    This approach preserves paragraph structure by trying to split on paragraph boundaries first.
    Creates Document objects from chunks and adds them to the vector store.
    Prints detailed statistics and comparison metrics.
    
    Args:
        vector_store: The vector store to add documents to
        file_path: Path to the document file to read and chunk
    
    Returns:
        Total number of chunks successfully stored
    """
    try:
        # Read the file
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Create RecursiveCharacterTextSplitter with paragraph-aware separators
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=0,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Create Document objects from the text
        chunks = splitter.create_documents([text])
        
        # Calculate statistics
        num_chunks = len(chunks)
        chunk_sizes = [len(chunk.page_content) for chunk in chunks]
        min_chunk_size = min(chunk_sizes) if chunk_sizes else 0
        max_chunk_size = max(chunk_sizes) if chunk_sizes else 0
        avg_chunk_size = len(text) / num_chunks if num_chunks > 0 else 0
        
        # Count chunks that start with newline (paragraph preservation indicator)
        chunks_starting_with_newline = sum(1 for chunk in chunks if chunk.page_content.startswith("\n"))
        
        print(f"📊 Recursive Chunking Statistics for '{os.path.basename(file_path)}':")
        print(f"   - Total chunks created: {num_chunks}")
        print(f"   - Smallest chunk size: {min_chunk_size} characters")
        print(f"   - Largest chunk size: {max_chunk_size} characters")
        print(f"   - Average chunk size: {avg_chunk_size:.1f} characters")
        print(f"   - Chunks starting with newline: {chunks_starting_with_newline} (paragraph preservation)")
        print()
        
        # Pass chunks to load_document_with_chunks
        return load_document_with_chunks(vector_store, file_path, chunks)
    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
        return 0
    except Exception as e:
        print(f"❌ Error in load_with_recursive_chunking for '{file_path}': {str(e)}")
        return 0

def load_with_markdown_structure_chunking(vector_store, file_path):
    """
    Reads a markdown file and splits it by header structure using MarkdownHeaderTextSplitter.
    Then applies RecursiveCharacterTextSplitter to maintain chunk size while preserving headers.
    This two-stage approach maximizes semantic coherence by respecting document structure.
    Creates Document objects from chunks and adds them to the vector store.
    Prints detailed statistics showing structure preservation.
    
    Args:
        vector_store: The vector store to add documents to
        file_path: Path to the markdown document file to read and chunk
    
    Returns:
        Total number of chunks successfully stored
    """
    try:
        # Read the file
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # First stage: Split by markdown headers to preserve structure
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2")
        ]
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        md_chunks = markdown_splitter.split_text(text)
        
        # Second stage: Apply recursive chunking with overlap to preserve context
        recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=8000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Split markdown chunks further if needed
        chunks = []
        for md_chunk in md_chunks:
            # Create Document from markdown chunk to preserve metadata (headers)
            doc = Document(page_content=md_chunk.page_content, metadata=md_chunk.metadata)
            # Split if needed while preserving metadata
            split_docs = recursive_splitter.split_documents([doc])
            chunks.extend(split_docs)
        
        # Calculate statistics
        num_chunks = len(chunks)
        chunk_sizes = [len(chunk.page_content) for chunk in chunks]
        min_chunk_size = min(chunk_sizes) if chunk_sizes else 0
        max_chunk_size = max(chunk_sizes) if chunk_sizes else 0
        avg_chunk_size = len(text) / num_chunks if num_chunks > 0 else 0
        
        # Count chunks with header metadata (structure preservation indicator)
        chunks_with_headers = sum(1 for chunk in chunks if any(key.startswith("Header") for key in chunk.metadata.keys()))
        
        print(f"📊 Markdown Structure Chunking Statistics for '{os.path.basename(file_path)}':")
        print(f"   - Total chunks created: {num_chunks}")
        print(f"   - Smallest chunk size: {min_chunk_size} characters")
        print(f"   - Largest chunk size: {max_chunk_size} characters")
        print(f"   - Average chunk size: {avg_chunk_size:.1f} characters")
        print(f"   - Chunks with header metadata: {chunks_with_headers} (structure preservation)")
        print(f"   - Overlap: 200 characters (context preservation across chunks)")
        print()
        
        # Pass chunks to load_document_with_chunks
        return load_document_with_chunks(vector_store, file_path, chunks)
    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
        return 0
    except Exception as e:
        print(f"❌ Error in load_with_markdown_structure_chunking for '{file_path}': {str(e)}")
        return 0

def create_search_tool(vector_store):
    """
    Creates a search tool that can be used by LangChain agents.
    The returned tool allows agents to query the company document repository.
    
    Args:
        vector_store: The vector store to search
    
    Returns:
        A LangChain Tool for searching documents
    """
    @tool
    def search_documents(query: str) -> str:
        """Searches the company document repository for relevant information based on the given query. Use this to find information about company policies, benefits, and procedures."""
        results = vector_store.similarity_search_with_score(query, k=3)
        
        formatted_results = []
        for idx, (doc, score) in enumerate(results, 1):
            formatted_results.append(f"Result {idx} (Score: {score:.4f}): {doc.page_content}")
        
        return "\n\n".join(formatted_results)
    
    return search_documents

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

    # Create ChatOpenAI instance for agent interactions
    chat_model = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        base_url="https://models.inference.ai.azure.com",
        api_key=os.getenv("GITHUB_TOKEN")
    )

    # Load documents into the vector database
    print("=== Loading Documents into Vector Database ===\n")
    file_to_load = "HealthInsuranceBrochure.md"
    doc_id = load_document(vector_store, file_to_load)
    if doc_id:
        print(f"Document '{file_to_load}' loaded successfully with ID: {doc_id}\n")
    else:
        print(f"Failed to load document: {file_to_load}\n")

    # Load EmployeeHandbook.md with markdown structure-aware chunking
    file_to_load2 = "EmployeeHandbook.md"
    chunks_stored = load_with_markdown_structure_chunking(vector_store, file_to_load2)
    if chunks_stored > 0:
        print(f"Document '{file_to_load2}' chunks loaded successfully\n")
    else:
        print(f"Failed to load document: {file_to_load2}\n")
    
    # Create the search tool for the agent
    search_tool = create_search_tool(vector_store)
    
    # Create the agent using LangChain's ReAct pattern with verbose output
    agent = create_agent(
        model=chat_model,
        tools=[search_tool],
        system_prompt=(
            "You are a helpful assistant that answers questions about company policies, benefits, and procedures. "
            "Use the search_documents tool to find relevant information before answering. "
            "Always cite which document chunks you used in your answer."
        ),
    
    )
    
    print("=== Agent Ready ===\n")
    print("Agent is initialized and ready to answer questions about company policies and procedures.\n")
    
    # Initialize chat history for conversation tracking
    chat_history = []
    
    # Welcome message
    print("📚 Welcome to the Company Policy Assistant!")
    print("━" * 60)
    print("I can help you find information about:")
    print("  • Company policies and procedures")
    print("  • Employee benefits and eligibility")
    print("  • Health insurance options")
    print("  • And more from company documents")
    print("\nType 'quit' or 'exit' to end the conversation.")
    print("━" * 60)
    print()
    
    # Interactive chat loop
    while True:
        # Get user input
        user_input = input("You: ").strip()
        
        # Check for exit commands
        if user_input.lower() in ["quit", "exit"]:
            print("\n👋 Thank you for using the Company Policy Assistant. Goodbye!")
            break
        
        # Skip empty input
        if not user_input:
            continue
        
        # Add user message to chat history
        chat_history.append(HumanMessage(content=user_input))
        
        # Invoke the agent with messages format
        result = agent.invoke({"messages": chat_history})
        
        # Extract the agent's response from the messages list
        if isinstance(result, dict) and "messages" in result:
            messages = result["messages"]
            # Get the last message (should be the agent's response)
            agent_response = messages[-1].content if messages else ""
        elif hasattr(result, "content"):
            agent_response = result.content
        else:
            agent_response = str(result)
        
        print(f"\nAgent: {agent_response}\n")
        
        # Add agent response to chat history
        chat_history.append(AIMessage(content=agent_response))

if __name__ == "__main__":
    main()