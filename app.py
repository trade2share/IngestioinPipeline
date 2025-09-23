import streamlit as st
import os
from dotenv import load_dotenv
from ingestion import load_documents, split_documents, chunk_and_vectorstore
import time

# Page configuration
st.set_page_config(
    page_title="Dokumenten-Indizierungs-Pipeline",
    page_icon="",
    layout="wide"
)

# Load environment variables
load_dotenv()

def main():
    st.title("Dokumenten-Indizierungs-Pipeline")
    st.markdown("---")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Konfiguration")
        
        # Check if environment variables are set
        conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        container = os.getenv("AZURE_STORAGE_CONTAINER_NAME")
        azure_openai_key = os.getenv("AZURE_OPENAI_API_KEY")
        azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        pinecone_key = os.getenv("PINECONE_API_KEY")
        pinecone_index = os.getenv("PINECONE_INDEX_NAME")
        
        if conn_str and container:
            st.success("Azure Storage konfiguriert")
            st.info(f"Container: {container}")
        else:
            st.error("Azure Storage nicht konfiguriert")
            st.warning("Bitte AZURE_STORAGE_CONNECTION_STRING und AZURE_STORAGE_CONTAINER_NAME in Ihrer .env-Datei setzen")
        
        if azure_openai_key and azure_openai_endpoint:
            st.success("Azure OpenAI konfiguriert")
            st.info(f"Endpoint: {azure_openai_endpoint[:50]}...")
        else:
            st.warning("Azure OpenAI nicht konfiguriert")
        
        if pinecone_key and pinecone_index:
            st.success("Pinecone konfiguriert")
            st.info(f"Index: {pinecone_index}")
        else:
            st.warning("Pinecone nicht konfiguriert")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Pipeline starten")
        st.markdown("""
        Die Pipeline wird:
        1. Dokumente aus Azure Blob Storage laden
        2. Dokumente in Chunks teilen
        3. Embeddings erstellen und in die Vektordatenbank speichern
        """)
        
        # Indexing button
        if st.button("Pipeline starten", type="primary", use_container_width=True):
            if not conn_str or not container:
                st.error("Indizierung kann nicht gestartet werden: Azure Storage nicht konfiguriert")
                return
            
            if not azure_openai_key or not azure_openai_endpoint:
                st.error("Indizierung kann nicht gestartet werden: Azure OpenAI nicht konfiguriert")
                return
            
            if not pinecone_key or not pinecone_index:
                st.error("Indizierung kann nicht gestartet werden: Pinecone nicht konfiguriert")
                return
            
            # Create progress containers
            progress_container = st.container()
            status_container = st.container()
            results_container = st.container()
            
            with progress_container:
                progress_bar = st.progress(0)
                status_text = st.empty()
            
            with status_container:
                st.info("🔄 Indizierungsprozess wird gestartet...")
            
            try:
                # Step 1: Load documents
                status_text.text("Dokumente werden aus Azure Blob Storage geladen...")
                progress_bar.progress(25)
                
                docs = load_documents()
                st.success(f"{len(docs)} Dokumente geladen")
                
                # Step 2: Split documents
                status_text.text("Dokumente werden in Chunks aufgeteilt...")
                progress_bar.progress(50)
                
                chunks = split_documents(docs)
                st.success(f"{len(chunks)} Chunks erstellt")
                
                # Step 3: Create embeddings and store in vector database
                status_text.text("Embeddings werden erstellt und in Vektordatenbank gespeichert...")
                progress_bar.progress(75)
                
                chunk_and_vectorstore(chunks)
                st.success("Dokumente in Vektordatenbank indiziert")
                
                # Step 4: Complete
                status_text.text("Indizierung abgeschlossen!")
                progress_bar.progress(100)
                
                # Final results
                with results_container:
                    st.success("Indizierung erfolgreich abgeschlossen!")
                    
                    # Display statistics
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Dokumente", len(docs))
                    with col_b:
                        st.metric("Chunks", len(chunks))
                    with col_c:
                        st.metric("Status", "✅ Abgeschlossen")
                    
                    # Show sample chunks
                    if chunks:
                        st.subheader("Beispiel-Chunks")
                        for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
                            with st.expander(f"Chunk {i+1} (Länge: {len(chunk.page_content)} Zeichen)"):
                                st.text(chunk.page_content[:500] + "..." if len(chunk.page_content) > 500 else chunk.page_content)
                
                status_text.text("✅ Indizierung abgeschlossen!")
                
            except Exception as e:
                st.error(f"❌ Fehler während der Indizierung: {str(e)}")
                status_text.text("❌ Indizierung fehlgeschlagen!")
    
    with col2:
        st.header("Pipeline-Status")
        
        # Check system status
        st.subheader("System-Status")
        
        # Check Azure connection
        if conn_str and container:
            st.success("Azure Storage: Verbunden")
        else:
            st.error("Azure Storage: Nicht konfiguriert")
        
        # Check required packages
        try:
            import langchain_community
            import langchain_openai
            import langchain_pinecone
            import pinecone
            st.success("Abhängigkeiten: Alle installiert")
        except ImportError as e:
            st.error(f"Abhängigkeiten: Fehlend - {str(e)}")
        
        # Check API keys
        if azure_openai_key and azure_openai_endpoint:
            st.success("Azure OpenAI: Konfiguriert")
        else:
            st.error("Azure OpenAI: Nicht konfiguriert")
        
        if pinecone_key and pinecone_index:
            st.success("Pinecone: Konfiguriert")
        else:
            st.error("Pinecone: Nicht konfiguriert")
        
        
        

if __name__ == "__main__":
    main()