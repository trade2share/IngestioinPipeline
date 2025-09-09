import streamlit as st
import os
from dotenv import load_dotenv
from ingestion import load_documents, split_documents, chunk_and_vectorstore
import time

# Page configuration
st.set_page_config(
    page_title="Document Indexing Pipeline",
    page_icon="📚",
    layout="wide"
)

# Load environment variables
load_dotenv()

def main():
    st.title("📚 Document Indexing Pipeline")
    st.markdown("---")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Check if environment variables are set
        conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        container = os.getenv("AZURE_STORAGE_CONTAINER_NAME")
        azure_openai_key = os.getenv("AZURE_OPENAI_API_KEY")
        azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        pinecone_key = os.getenv("PINECONE_API_KEY")
        pinecone_index = os.getenv("PINECONE_INDEX_NAME")
        
        if conn_str and container:
            st.success("✅ Azure Storage configured")
            st.info(f"Container: {container}")
        else:
            st.error("❌ Azure Storage not configured")
            st.warning("Please set AZURE_STORAGE_CONNECTION_STRING and AZURE_STORAGE_CONTAINER_NAME in your .env file")
        
        if azure_openai_key and azure_openai_endpoint:
            st.success("✅ Azure OpenAI configured")
            st.info(f"Endpoint: {azure_openai_endpoint[:50]}...")
        else:
            st.warning("⚠️ Azure OpenAI not configured")
        
        if pinecone_key and pinecone_index:
            st.success("✅ Pinecone configured")
            st.info(f"Index: {pinecone_index}")
        else:
            st.warning("⚠️ Pinecone not configured")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🚀 Start Indexing")
        st.markdown("""
        This pipeline will:
        1. Load documents from Azure Blob Storage
        2. Process and split documents into chunks
        3. Create embeddings and store in vector database
        """)
        
        # Indexing button
        if st.button("🔄 Start Indexing", type="primary", use_container_width=True):
            if not conn_str or not container:
                st.error("❌ Cannot start indexing: Azure Storage not configured")
                return
            
            if not azure_openai_key or not azure_openai_endpoint:
                st.error("❌ Cannot start indexing: Azure OpenAI not configured")
                return
            
            if not pinecone_key or not pinecone_index:
                st.error("❌ Cannot start indexing: Pinecone not configured")
                return
            
            # Create progress containers
            progress_container = st.container()
            status_container = st.container()
            results_container = st.container()
            
            with progress_container:
                progress_bar = st.progress(0)
                status_text = st.empty()
            
            with status_container:
                st.info("🔄 Starting indexing process...")
            
            try:
                # Step 1: Load documents
                status_text.text("📥 Loading documents from Azure Blob Storage...")
                progress_bar.progress(25)
                
                docs = load_documents()
                st.success(f"✅ Loaded {len(docs)} documents")
                
                # Step 2: Split documents
                status_text.text("✂️ Splitting documents into chunks...")
                progress_bar.progress(50)
                
                chunks = split_documents(docs)
                st.success(f"✅ Created {len(chunks)} chunks")
                
                # Step 3: Create embeddings and store in vector database
                status_text.text("🧠 Creating embeddings and storing in vector database...")
                progress_bar.progress(75)
                
                chunk_and_vectorstore(chunks)
                st.success("✅ Documents indexed in vector database")
                
                # Step 4: Complete
                status_text.text("✅ Indexing completed!")
                progress_bar.progress(100)
                
                # Final results
                with results_container:
                    st.success("🎉 Indexing completed successfully!")
                    
                    # Display statistics
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Documents", len(docs))
                    with col_b:
                        st.metric("Chunks", len(chunks))
                    with col_c:
                        st.metric("Status", "✅ Complete")
                    
                    # Show sample chunks
                    if chunks:
                        st.subheader("📄 Sample Chunks")
                        for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
                            with st.expander(f"Chunk {i+1} (Length: {len(chunk.page_content)} chars)"):
                                st.text(chunk.page_content[:500] + "..." if len(chunk.page_content) > 500 else chunk.page_content)
                
                status_text.text("✅ Indexing completed!")
                
            except Exception as e:
                st.error(f"❌ Error during indexing: {str(e)}")
                status_text.text("❌ Indexing failed!")
    
    with col2:
        st.header("📊 Pipeline Status")
        
        # Check system status
        st.subheader("System Status")
        
        # Check Azure connection
        if conn_str and container:
            st.success("🟢 Azure Storage: Connected")
        else:
            st.error("🔴 Azure Storage: Not configured")
        
        # Check required packages
        try:
            import langchain_community
            import langchain_openai
            import langchain_pinecone
            import pinecone
            st.success("🟢 Dependencies: All installed")
        except ImportError as e:
            st.error(f"🔴 Dependencies: Missing - {str(e)}")
        
        # Check API keys
        if azure_openai_key and azure_openai_endpoint:
            st.success("🟢 Azure OpenAI: Configured")
        else:
            st.error("🔴 Azure OpenAI: Not configured")
        
        if pinecone_key and pinecone_index:
            st.success("🟢 Pinecone: Configured")
        else:
            st.error("🔴 Pinecone: Not configured")
        
        st.subheader("📈 Statistics")
        st.info("Run indexing to see statistics")
        
        # Instructions
        st.subheader("📋 Instructions")
        st.markdown("""
        1. Ensure your `.env` file contains:
           - `AZURE_STORAGE_CONNECTION_STRING`
           - `AZURE_STORAGE_CONTAINER_NAME`
           - `AZURE_OPENAI_API_KEY`
           - `AZURE_OPENAI_ENDPOINT`
           - `PINECONE_API_KEY`
           - `PINECONE_INDEX_NAME`
        2. Click "Start Indexing" to begin
        3. Monitor progress in real-time
        4. View results and statistics
        """)

if __name__ == "__main__":
    main()
