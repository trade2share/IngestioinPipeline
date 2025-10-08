import os
import time
from dotenv import load_dotenv
import re
import unicodedata

from langchain_community.document_loaders import AzureBlobStorageContainerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from pydantic import SecretStr
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain_core.documents import Document
from openai import RateLimitError

import tempfile

from langchain_community.document_loaders import PyPDFium2Loader
from azure.storage.blob import BlobServiceClient



def sanitize_text(text: str) -> str:
    """Normalize and clean text extracted from PDFs.

    Fixes common issues:
    - Compose Unicode to NFC so that combining marks become single codepoints (e.g., a + ¨ -> ä)
    - Remove soft hyphens and join hyphenated line breaks
    - Normalize non-breaking spaces
    - Convert spacing diaeresis placed before/after vowels into proper umlauts
    """
    if not text:
        return ""

    # Normalize to NFC first (handles combining marks like a + U+0308)
    cleaned = unicodedata.normalize("NFC", text)

    # Remove zero-width characters that often appear in PDF extraction
    cleaned = re.sub(r"[\u200B-\u200D\uFEFF]", "", cleaned)

    # Remove soft hyphens and fix hyphenation across line breaks
    cleaned = cleaned.replace("\u00AD", "")
    cleaned = re.sub(r"(\w+)-\n(\w+)", r"\1\2", cleaned)

    # Normalize non-breaking spaces to regular spaces
    cleaned = cleaned.replace("\u00A0", " ")

    # Map for German umlauts when diaeresis is extracted as a separate spacing char
    umlaut_map = {"a": "ä", "o": "ö", "u": "ü", "A": "Ä", "O": "Ö", "U": "Ü"}

    # Case 1: space + diaeresis before the vowel → replace with umlaut vowel
    cleaned = re.sub(r"\s[\u00A8¨]\s?([AOUaou])", lambda m: umlaut_map.get(m.group(1), m.group(1)), cleaned)

    # Case 2: vowel followed by space + diaeresis → replace with umlaut vowel
    cleaned = re.sub(r"([AOUaou])\s[\u00A8¨]", lambda m: umlaut_map.get(m.group(1), m.group(1)), cleaned)

    # Case 3: combining diaeresis U+0308 adjacent to vowel with optional space
    cleaned = re.sub(r"([AOUaou])\s*\u0308", lambda m: umlaut_map.get(m.group(1), m.group(1)), cleaned)
    cleaned = re.sub(r"\u0308\s*([AOUaou])", lambda m: umlaut_map.get(m.group(1), m.group(1)), cleaned)

    # Attempt to fix common UTF-8/CP1252 mojibake (e.g., Ã¤ → ä)
    if any(ch in cleaned for ch in ("Ã", "Â", "â")):
        def _reduce_mojibake(s: str) -> str:
            score = s.count("Ã") + s.count("Â") + s.count("â")
            for enc in ("cp1252", "latin1"):
                try:
                    cand = s.encode(enc, errors="ignore").decode("utf-8", errors="ignore")
                    cand_score = cand.count("Ã") + cand.count("Â") + cand.count("â")
                    if cand_score < score:
                        s, score = cand, cand_score
                except Exception:
                    pass
            return s
        cleaned = _reduce_mojibake(cleaned)

    # Compose again to be safe
    cleaned = unicodedata.normalize("NFC", cleaned)

    # Collapse multiple spaces introduced by cleanup
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned

load_dotenv()

def load_env():
    load_dotenv()
    return os.getenv("AZURE_STORAGE_CONNECTION_STRING"), os.getenv("AZURE_STORAGE_CONTAINER_NAME")

def load_azure_openai() -> tuple[str, str, str | None, str]:
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION")
    # Der Deployment-Name für das Embedding-Modell
    deployment_name = "text-embedding-3-small" 
    
    if not all([api_key, azure_endpoint, deployment_name]):
        raise ValueError("Bitte stellen Sie sicher, dass AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT und der Deployment-Name gesetzt sind.")
    
    # At this point, mypy/pyright can treat these as non-optional
    azure_api_key: str = api_key  # type: ignore[assignment]
    azure_endpoint_str: str = azure_endpoint  # type: ignore[assignment]
    return azure_api_key, azure_endpoint_str, api_version, deployment_name



def load_documents():
    conn_str, container_name = load_env()
    if not conn_str or not container_name:
        raise ValueError("AZURE_STORAGE_CONNECTION_STRING and AZURE_STORAGE_CONTAINER_NAME must be set")

    print("Stelle Verbindung zum Azure Blob Storage her...")
    blob_service_client = BlobServiceClient.from_connection_string(conn_str)
    container_client = blob_service_client.get_container_client(container_name)

    all_docs = []

    for blob in container_client.list_blobs():
        if not blob.name.lower().endswith(".pdf"):
            print(f" überspringe Datei (kein PDF): {blob.name}")
            continue

        print(f"Verarbeite PDF: {blob.name}...")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            try:
                blob_client = container_client.get_blob_client(blob.name)
                downloader = blob_client.download_blob()
                temp_file.write(downloader.readall())
                temp_file_path = temp_file.name

                loader = PyPDFium2Loader(temp_file_path)
                pages = loader.load()
                
                # ✨ DAS I-TÜPFELCHEN: KORRIGIERE DIE QUELLE ✨
                # Gehe durch jede geladene Seite und ersetze die temporäre Quelle
                # durch den echten Dateinamen aus Azure.
                for page in pages:
                    # Sanitize extracted text to fix umlauts and spacing issues
                    page.page_content = sanitize_text(page.page_content)
                    page.metadata["source"] = blob.name
                
                all_docs.extend(pages)
                print(f"{blob.name} erfolgreich geladen ({len(pages)} Seiten mit korrekter Quelle).")

            finally:
                os.remove(temp_file.name)

    return all_docs


def split_documents(docs: list[Document]) -> list[Document]:
    """
    Teilt Dokumente in Chunks und reichert die Metadaten jedes Chunks an.
    """
    # 1. Dokumente aufteilen
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    
    print(f"Dokumente in {len(chunks)} Chunks aufgeteilt. Beginne Anreicherung...")

    # 2. Durch die Chunks iterieren und ihre Metadaten direkt bearbeiten
    for i, chunk in enumerate(chunks):
        # Ensure each chunk's content is sanitized (in case of external loaders)
        chunk.page_content = sanitize_text(chunk.page_content)
        source = chunk.metadata.get('source', 'Unbekannte Quelle')
        source_filename = os.path.basename(source)
        page_num = chunk.metadata.get('page', chunk.metadata.get('page_number', 0))
        
        # Metadaten direkt im Chunk-Objekt aktualisieren
        chunk.metadata['source_filename'] = source_filename
        chunk.metadata['page_number'] = page_num + 1 if isinstance(page_num, int) else page_num
        
        # Eindeutige ID für den Chunk erstellen
        chunk.metadata['chunk_id'] = f"{source_filename}_seite-{chunk.metadata['page_number']}_chunk-{i}"

    print("Anreicherung der Metadaten abgeschlossen.")
    return chunks

def chunk_and_vectorstore(chunks, batch_size=15, delay_between_batches=1):
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

    # --- HINZUGEFÜGTER CODE ZUM ZURÜCKSETZEN DES INDEX ---
    print(f"🧹 Prüfe Index '{index_name}' auf vorhandene Vektoren...")
    try:
        # Prüfe, ob der Index Vektoren enthält
        stats = index.describe_index_stats()
        total_vector_count = stats.get('total_vector_count', 0)
        
        if total_vector_count > 0:
            print(f"Index enthält {total_vector_count} Vektoren. Starte Zurücksetzung...")
            index.delete(delete_all=True)
            print(f"Index '{index_name}' wurde erfolgreich zurückgesetzt.")
            # Eine kurze Pause, um sicherzustellen, dass der Löschvorgang serverseitig abgeschlossen ist.
            time.sleep(5)
        else:
            print(f"Index '{index_name}' ist bereits leer. Keine Zurücksetzung erforderlich.")
            
    except Exception as e:
        print(f"Fehler beim Zurücksetzen des Index: {e}")
        # Beendet die Funktion, wenn das Leeren fehlschlägt, um inkonsistente Daten zu vermeiden.
        return 0
    # --- ENDE DES HINZUGEFÜGTEN CODES ---

    vectorstore = PineconeVectorStore(index=index, embedding=embeddings)
    
    total_chunks = len(chunks)
    processed_chunks = 0
    
    print(f"Starte die Stapelverarbeitung von {total_chunks} Chunks...")
    print(f"Batch-Größe: {batch_size}, Verzögerung: {delay_between_batches}s")
    
    # Process chunks in batches
    for i in range(0, total_chunks, batch_size):
        batch = chunks[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (total_chunks + batch_size - 1) // batch_size
        
        print(f"\n Verarbeite Batch {batch_num}/{total_batches} ({len(batch)} Chunks)")
        
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # Process the batch
                vectorstore.add_documents(batch)
                processed_chunks += len(batch)
                print(f"Batch {batch_num} erfolgreich abgeschlossen ({processed_chunks}/{total_chunks} gesamt)")
                break
                
            except RateLimitError as e:
                retry_count += 1
                wait_time = 60 * retry_count  # Exponential backoff
                print(f"  Rate-Limit erreicht! Warte {wait_time}s vor Wiederholung {retry_count}/{max_retries}")
                time.sleep(wait_time)
                
            except Exception as e:
                print(f"Fehler in Batch {batch_num}: {str(e)}")
                retry_count += 1
                if retry_count >= max_retries:
                    print(f"Fehler bei der Verarbeitung von Batch {batch_num} nach {max_retries} Versuchen")
                    break
                time.sleep(5)
        
        # Wait between batches (except for the last batch)
        if i + batch_size < total_chunks:
            print(f"Warte {delay_between_batches}s vor dem nächsten Batch...")
            time.sleep(delay_between_batches)
    
    print(f"\n Stapelverarbeitung abgeschlossen!")
    print(f" Erfolgreich verarbeitet: {processed_chunks}/{total_chunks} Chunks")
    
    return processed_chunks


if __name__ == "__main__":
    docs = load_documents()
    chunks = split_documents(docs)

    chunk_and_vectorstore(chunks)

    