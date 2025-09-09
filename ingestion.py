import os
import time
from dotenv import load_dotenv

from langchain_community.document_loaders import AzureBlobStorageContainerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from pydantic import SecretStr
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain_core.documents import Document
from openai import RateLimitError



def load_env():
    load_dotenv()
    return os.getenv("AZURE_STORAGE_CONNECTION_STRING"), os.getenv("AZURE_STORAGE_CONTAINER_NAME")

def load_azure_openai():
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION")
    # Der Deployment-Name für das Embedding-Modell
    deployment_name = "text-embedding-3-small" 
    
    if not all([api_key, azure_endpoint, deployment_name]):
        raise ValueError("Bitte stellen Sie sicher, dass AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT und der Deployment-Name gesetzt sind.")
    
    return api_key, azure_endpoint, api_version, deployment_name



def load_documents():
    conn_str, container = load_env()
    if not conn_str or not container:
        raise ValueError("AZURE_STORAGE_CONNECTION_STRING and AZURE_STORAGE_CONTAINER_NAME must be set")
    
    loader = AzureBlobStorageContainerLoader(
        conn_str=conn_str, container=container)
    
    # Load documents from Azure Blob Storage
    docs = loader.load()
    
    return docs


def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    return chunks

def chunk_and_vectorstore(chunks, batch_size=5, delay_between_batches=2):
    """
    Process chunks in batches to avoid rate limits
    
    Args:
        chunks: List of document chunks
        batch_size: Number of chunks to process at once
        delay_between_batches: Seconds to wait between batches
    """
    
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME")
    
    if not api_key or not index_name:
        raise ValueError("PINECONE_API_KEY and PINECONE_INDEX_NAME must be set")
    
    # Load Azure OpenAI configuration
    azure_api_key, azure_endpoint, azure_api_version, deployment_name = load_azure_openai()
    
    # Create Azure OpenAI embeddings
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=deployment_name,
        azure_endpoint=azure_endpoint,
        api_key=SecretStr(azure_api_key),
        api_version=azure_api_version,
        openai_api_type="azure"
    )

    pc = Pinecone(api_key=api_key)
    index = pc.Index(name=index_name)
    vectorstore = PineconeVectorStore(index=index, embedding=embeddings)
    
    total_chunks = len(chunks)
    processed_chunks = 0
    
    print(f"🚀 Starting batch processing of {total_chunks} chunks...")
    print(f"📦 Batch size: {batch_size}, Delay: {delay_between_batches}s")
    
    # Process chunks in batches
    for i in range(0, total_chunks, batch_size):
        batch = chunks[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (total_chunks + batch_size - 1) // batch_size
        
        print(f"\n📦 Processing batch {batch_num}/{total_batches} ({len(batch)} chunks)")
        
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # Process the batch
                vectorstore.add_documents(batch)
                processed_chunks += len(batch)
                print(f"✅ Batch {batch_num} completed successfully ({processed_chunks}/{total_chunks} total)")
                break
                
            except RateLimitError as e:
                retry_count += 1
                wait_time = 60 * retry_count  # Exponential backoff
                print(f"⚠️  Rate limit hit! Waiting {wait_time}s before retry {retry_count}/{max_retries}")
                time.sleep(wait_time)
                
            except Exception as e:
                print(f"❌ Error in batch {batch_num}: {str(e)}")
                retry_count += 1
                if retry_count >= max_retries:
                    print(f"💥 Failed to process batch {batch_num} after {max_retries} retries")
                    break
                time.sleep(5)
        
        # Wait between batches (except for the last batch)
        if i + batch_size < total_chunks:
            print(f"⏳ Waiting {delay_between_batches}s before next batch...")
            time.sleep(delay_between_batches)
    
    print(f"\n🎉 Batch processing completed!")
    print(f"📊 Successfully processed: {processed_chunks}/{total_chunks} chunks")
    
    return processed_chunks




if __name__ == "__main__":
    docs = load_documents()
    chunks = split_documents(docs)
    
    print(f"Anzahl der Chunks: {len(chunks)}\n")

    chunk_and_vectorstore(chunks)