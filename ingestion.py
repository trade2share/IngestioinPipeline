import os
import time
from dotenv import load_dotenv
import re
import unicodedata
import tempfile
from typing import cast

from langchain_community.document_loaders import PyPDFium2Loader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from pydantic import SecretStr
from langchain_core.documents import Document
from openai import RateLimitError
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.storage.blob import BlobServiceClient

# Lädt Umgebungsvariablen aus der .env-Datei
load_dotenv()

def sanitize_text(text: str) -> str:
    """
    Normalisiert und bereinigt aus PDFs extrahierten Text.
    Behebt häufige Probleme wie falsche Umlaute, Zeilenumbrüche und Leerzeichen.
    """
    if not text:
        return ""

    # Unicode-Normalisierung (NFC)
    cleaned = unicodedata.normalize("NFC", text)
    # Zeichen ohne Breite entfernen
    cleaned = re.sub(r"[\u200B-\u200D\uFEFF]", "", cleaned)
    # Weiche Trennstriche und Zeilenumbrüche korrigieren
    cleaned = cleaned.replace("\u00AD", "")
    cleaned = re.sub(r"(\w+)-\n(\w+)", r"\1\2", cleaned)
    # Geschützte Leerzeichen normalisieren
    cleaned = cleaned.replace("\u00A0", " ")

    # Deutsche Umlaute aus separaten Diakritika wiederherstellen
    umlaut_map = {"a": "ä", "o": "ö", "u": "ü", "A": "Ä", "O": "Ö", "U": "Ü"}
    cleaned = re.sub(r"\s?¨\s?([AOUaou])", lambda m: umlaut_map.get(m.group(1), m.group(1)), cleaned)
    cleaned = re.sub(r"([AOUaou])\s?¨", lambda m: umlaut_map.get(m.group(1), m.group(1)), cleaned)
    
    # Häufige Kodierungsfehler (Mojibake) korrigieren
    try:
        cleaned_bytes = cleaned.encode('latin1')
        fixed_text = cleaned_bytes.decode('utf8')
        if 'Ã' not in fixed_text: # Prüfen, ob die Korrektur sinnvoll war
            cleaned = fixed_text
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass # Wenn die Konvertierung fehlschlägt, Originaltext beibehalten

    # Mehrfache Leerzeichen reduzieren
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned


def sanitize_document_key(value: str) -> str:
    """Sanitizes a string to a valid Azure Search document key.
    Allowed: letters, digits, underscore (_), dash (-), equal sign (=).
    """
    if value is None:
        value = ""
    # Remove accents/diacritics and normalize
    normalized = unicodedata.normalize("NFKD", value)
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    # Replace any disallowed char with underscore
    sanitized = re.sub(r"[^A-Za-z0-9_\-=]", "_", normalized)
    # Collapse multiple underscores and trim
    sanitized = re.sub(r"_+", "_", sanitized).strip("_")
    # Ensure non-empty
    if not sanitized:
        sanitized = "doc"
    return sanitized


def load_env():
    """Lädt Blob Storage Umgebungsvariablen."""
    return os.getenv("AZURE_STORAGE_CONNECTION_STRING"), os.getenv("AZURE_STORAGE_CONTAINER_NAME")

def load_azure_openai() -> tuple[str, str, str | None, str]:
    """Lädt Azure OpenAI Umgebungsvariablen."""
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION")
    deployment_name = "text-embedding-3-small" 
    
    if not all([api_key, azure_endpoint, deployment_name]):
        raise ValueError("Bitte stellen Sie sicher, dass AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT und der Deployment-Name gesetzt sind.")
    
    # Nach Validierung sind api_key und azure_endpoint Strings
    api_key = cast(str, api_key)
    azure_endpoint = cast(str, azure_endpoint)
    return api_key, azure_endpoint, api_version, deployment_name

def _load_azure_ai_search_env() -> tuple[str, str, str]:
    """Lädt Azure AI Search Umgebungsvariablen."""
    service_name = os.getenv("AZURE_AI_SEARCH_SERVICE_NAME")
    index_name = os.getenv("AZURE_AI_SEARCH_INDEX_NAME")
    api_key = os.getenv("AZURE_AI_SEARCH_API_KEY")

    if not service_name or not index_name or not api_key:
        raise ValueError(
            "AZURE_AI_SEARCH_SERVICE_NAME, AZURE_AI_SEARCH_INDEX_NAME und AZURE_AI_SEARCH_API_KEY müssen gesetzt sein"
        )
    return service_name, index_name, api_key


def load_documents() -> list[Document]:
    """Lädt PDF-Dokumente aus einem Azure Blob Storage Container."""
    conn_str, container_name = load_env()
    if not conn_str or not container_name:
        raise ValueError("AZURE_STORAGE_CONNECTION_STRING and AZURE_STORAGE_CONTAINER_NAME must be set")

    print("Stelle Verbindung zum Azure Blob Storage her...")
    blob_service_client = BlobServiceClient.from_connection_string(conn_str)
    container_client = blob_service_client.get_container_client(container_name)
    
    all_docs = []
    blobs = list(container_client.list_blobs())
    print(f"{len(blobs)} Blobs im Container gefunden.")

    for blob in blobs:
        if not blob.name.lower().endswith(".pdf"):
            print(f"  -> Überspringe (kein PDF): {blob.name}")
            continue

        print(f"Verarbeite PDF: {blob.name}...")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file_path = temp_file.name
            try:
                blob_client = container_client.get_blob_client(blob.name)
                downloader = blob_client.download_blob()
                temp_file.write(downloader.readall())
                temp_file.close()

                loader = PyPDFium2Loader(temp_file_path)
                pages = loader.load()
                
                for page in pages:
                    page.page_content = sanitize_text(page.page_content)
                    page.metadata["source"] = blob.name
                
                all_docs.extend(pages)
                print(f"  -> {blob.name} erfolgreich geladen ({len(pages)} Seiten).")

            finally:
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)

    return all_docs


def split_documents(docs: list[Document]) -> list[Document]:
    """Teilt Dokumente in Chunks und reichert die Metadaten an."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    
    print(f"Dokumente in {len(chunks)} Chunks aufgeteilt. Beginne Anreicherung...")

    for i, chunk in enumerate(chunks):
        source = chunk.metadata.get('source', 'Unbekannte Quelle')
        page_num = chunk.metadata.get('page', 0)
        
        chunk.metadata['source_filename'] = os.path.basename(source)
        chunk.metadata['page_number'] = page_num + 1
        raw_chunk_id = f"{chunk.metadata['source_filename']}_seite-{chunk.metadata['page_number']}_chunk-{i}"
        chunk_id = sanitize_document_key(raw_chunk_id)
        chunk.metadata['chunk_id'] = chunk_id
        # Setze eine stabile Dokument-ID gleich dem Chunk-Identifier
        chunk.metadata['id'] = chunk_id
        try:
            # Neuere LangChain-Versionen unterstützen Document.id für persistente IDs
            chunk.id = chunk_id  # type: ignore[attr-defined]
        except Exception:
            pass

    print("Anreicherung der Metadaten abgeschlossen.")
    return chunks


def _assert_azure_search_packages() -> None:
    """Stellt sicher, dass die Azure AI Search SDK-Pakete importierbar sind."""
    try:
        from azure.search.documents import SearchClient
        from azure.core.credentials import AzureKeyCredential
    except ImportError as e:
        raise ImportError(
            "Azure AI Search SDK nicht gefunden. Bitte installieren Sie es mit: "
            "`pip install -r requirements.txt`"
        ) from e


def chunk_and_vectorstore(chunks: list[Document], batch_size=16, delay_between_batches=2):
    """Erstellt Embeddings und speichert sie in Azure AI Search (ohne 'metadata'-Feld)."""
    _assert_azure_search_packages()

    service_name, index_name, search_api_key = _load_azure_ai_search_env()
    openai_api_key, openai_endpoint, openai_api_version, openai_deployment = load_azure_openai()
    
    endpoint = f"https://{service_name}.search.windows.net"
    
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=openai_deployment,
        azure_endpoint=openai_endpoint,
        api_key=SecretStr(openai_api_key),
        api_version=openai_api_version,
    )

    search_client = SearchClient(
        endpoint=endpoint,
        index_name=index_name,
        credential=AzureKeyCredential(search_api_key),
    )
    
    total_chunks = len(chunks)
    processed_chunks = 0
    
    print(f"\nStarte die Stapelverarbeitung von {total_chunks} Chunks...")
    
    for i in range(0, total_chunks, batch_size):
        batch = chunks[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        
        print(f"  Verarbeite Batch {batch_num} ({len(batch)} Chunks)...")
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Embeddings für alle Texte im Batch berechnen
                texts = [doc.page_content for doc in batch]
                vectors = embeddings.embed_documents(texts)

                # Dokumente explizit auf das Indexschema mappen (ohne 'metadata')
                docs_to_upload = []
                for doc, vec in zip(batch, vectors):
                    meta = doc.metadata or {}
                    doc_id = meta.get('id') or meta.get('chunk_id')
                    if not doc_id:
                        # Fallback: deterministische ID aus Quelle/Seite/Position
                        source_fn = meta.get('source_filename') or meta.get('source') or 'unknown'
                        page_no = meta.get('page_number', 0)
                        doc_id = f"{source_fn}_seite-{page_no}_auto"
                    # Azure Search Dokument-Key sanitizen (nur erlaubte Zeichen)
                    doc_id = sanitize_document_key(str(doc_id))
                    docs_to_upload.append({
                        'id': doc_id,
                        'content': doc.page_content,
                        'embedding': vec,
                        'source': meta.get('source_filename') or meta.get('source') or '',
                        'page_number': int(meta.get('page_number', 0)),
                        'chunk_id': meta.get('chunk_id') or doc_id,
                    })

                results = search_client.upload_documents(documents=docs_to_upload)
                failed = [r for r in results if not r.succeeded]
                if failed:
                    raise Exception(f"{len(failed)} Dokument(e) fehlgeschlagen, z.B. {failed[0].error_message}")

                processed_chunks += len(batch)
                print(f"  -> Batch {batch_num} erfolgreich abgeschlossen ({processed_chunks}/{total_chunks} gesamt)")
                break
            except RateLimitError:
                wait_time = 60 * (attempt + 1)
                print(f"  -> Rate-Limit erreicht! Warte {wait_time}s (Versuch {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
            except Exception as e:
                print(f"  -> Fehler in Batch {batch_num}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(5)
                else:
                    print(f"  -> FEHLER: Batch {batch_num} konnte nach {max_retries} Versuchen nicht verarbeitet werden.")
                    break
        
        if i + batch_size < total_chunks:
            time.sleep(delay_between_batches)
    
    print(f"\nStapelverarbeitung abgeschlossen! {processed_chunks}/{total_chunks} Chunks verarbeitet.")
    return processed_chunks


if __name__ == "__main__":
    try:
        docs = load_documents()
        if not docs:
            print("Keine Dokumente zum Verarbeiten gefunden. Das Skript wird beendet.")
        else:
            chunks = split_documents(docs)
            chunk_and_vectorstore(chunks)
            
    except ValueError as e:
        print(f"Konfigurationsfehler: {e}")
    except Exception as e:
        print(f"Ein unerwarteter Fehler ist aufgetreten: {e}")
    