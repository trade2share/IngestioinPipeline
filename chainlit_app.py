import os
from typing import List

import chainlit as cl
from dotenv import load_dotenv

from ingestion import load_documents, split_documents, chunk_and_vectorstore
from langchain_core.documents import Document


def get_env_status_markdown() -> str:
    """Compose a Markdown status overview of required configuration."""
    conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    container = os.getenv("AZURE_STORAGE_CONTAINER_NAME")
    azure_openai_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_ai_search_service = os.getenv("AZURE_AI_SEARCH_SERVICE_NAME")
    azure_ai_search_index = os.getenv("AZURE_AI_SEARCH_INDEX_NAME")
    azure_ai_search_api_key = os.getenv("AZURE_AI_SEARCH_API_KEY")

    status_lines = [
        "### Konfiguration",
        f" **Azure Storage**: {' konfiguriert' if conn_str and container else ' fehlt'}",
        f"   Container: `{container}`" if container else "  - Container: `-`",
        f" **Azure OpenAI**: {' konfiguriert' if azure_openai_key and azure_openai_endpoint else ' fehlt'}",
        f"  Endpoint: `{(azure_openai_endpoint[:50] + '...') if azure_openai_endpoint else '-'}`",
        f" **Azure AI Search**: {' konfiguriert' if azure_ai_search_service and azure_ai_search_index else ' fehlt'}",
        f"   Service: `{azure_ai_search_service or '-'}`",
        f"   Index: `{azure_ai_search_index or '-'}`",
    ]
    return "\n".join(status_lines)


def validate_required_env() -> tuple[bool, str]:
    """Validate presence of required environment variables and return (ok, error_msg)."""
    conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    container = os.getenv("AZURE_STORAGE_CONTAINER_NAME")
    azure_openai_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_ai_search_service = os.getenv("AZURE_AI_SEARCH_SERVICE_NAME")
    azure_ai_search_index = os.getenv("AZURE_AI_SEARCH_INDEX_NAME")
    azure_ai_search_api_key = os.getenv("AZURE_AI_SEARCH_API_KEY")

    if not conn_str or not container:
        return False, "Azure Storage nicht konfiguriert (AZURE_STORAGE_CONNECTION_STRING, AZURE_STORAGE_CONTAINER_NAME)"
    if not azure_openai_key or not azure_openai_endpoint:
        return False, "Azure OpenAI nicht konfiguriert (AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT)"
    if not azure_ai_search_service or not azure_ai_search_index or not azure_ai_search_api_key:
        return False, "Azure AI Search nicht konfiguriert (AZURE_AI_SEARCH_SERVICE_NAME, AZURE_AI_SEARCH_INDEX_NAME, AZURE_AI_SEARCH_API_KEY)"
    return True, ""


def format_metrics_markdown(num_docs: int, num_chunks: int) -> str:
    return (
        " **Ergebnisse**\n"
        f"- **Dokumente**: {num_docs}\n"
        f"- **Chunks**: {num_chunks}\n"
        f"- **Status**:  Abgeschlossen"
    )


def truncate_text(text: str, max_len: int = 500) -> str:
    if text is None:
        return ""
    return text if len(text) <= max_len else text[:max_len] + "..."


@cl.on_chat_start
async def start_chat():
    """Initial chat entrypoint that mirrors the Streamlit dashboard overview."""
    load_dotenv()

    await cl.Message(content="## Dokumenten-Indizierungs-Pipeline\n---").send()

    # Show configuration status
    await cl.Message(content=get_env_status_markdown()).send()

    # Present an action to start the pipeline
    await cl.Message(
        content=(
            "### Pipeline starten\n"
            "Die Pipeline wird:\n"
            "1. Dokumente aus Azure Blob Storage laden\n"
            "2. Dokumente in Chunks teilen\n"
            "3. Embeddings erstellen und in die Vektordatenbank speichern\n"
        ),
        actions=[
            cl.Action(
                name="start_pipeline",
                label="Pipeline starten",
                value="start",
                payload={"command": "start"},
                description="Indizierung starten",
            ),
        ],
    ).send()


@cl.action_callback("start_pipeline")
async def on_start_pipeline(action: cl.Action):
    """Run the indexing pipeline with step-wise updates and results summary."""
    ok, error_msg = validate_required_env()
    if not ok:
        await cl.Message(content=f" Indizierung kann nicht gestartet werden: {error_msg}").send()
        return

    # Progress message we update as we go
    progress_msg = await cl.Message(content="🔄 Indizierungsprozess wird gestartet...").send()

    try:
        # Step 1: Load documents
        async with cl.Step(name="Dokumente laden"):
            progress_msg.content = "Dokumente werden aus Azure Blob Storage geladen... (25%)"
            await progress_msg.update()
            docs: List[Document] = load_documents()
            await cl.Message(content=f" {len(docs)} Dokumente geladen").send()

        # Step 2: Split documents
        async with cl.Step(name="Dokumente splitten"):
            progress_msg.content = "Dokumente werden in Chunks aufgeteilt... (50%)"
            await progress_msg.update()
            chunks: List[Document] = split_documents(docs)
            await cl.Message(content=f" {len(chunks)} Chunks erstellt").send()

        # Step 3: Create embeddings and store in vector DB
        async with cl.Step(name="Embeddings und Indexierung"):
            progress_msg.content = "Embeddings werden erstellt und in Vektordatenbank gespeichert... (75%)"
            await progress_msg.update()
            _processed = chunk_and_vectorstore(chunks)
            await cl.Message(content=" Dokumente in Vektordatenbank indiziert").send()

        # Complete
        progress_msg.content = " Indizierung abgeschlossen! (100%)"
        await progress_msg.update()

        # Results: metrics
        await cl.Message(content=format_metrics_markdown(num_docs=len(docs), num_chunks=len(chunks))).send()

        # Results: sample chunks
        if chunks:
            await cl.Message(content="### Beispiel-Chunks").send()
            sample_count = min(3, len(chunks))
            for i in range(sample_count):
                chunk = chunks[i]
                title = f"Chunk {i + 1} (Länge: {len(chunk.page_content)} Zeichen)"
                content = truncate_text(chunk.page_content)
                await cl.Message(content=f"**{title}**\n\n" + content).send()

    except Exception as e:  # noqa: BLE001 - report error to user
        progress_msg.content = " Indizierung fehlgeschlagen!"
        await progress_msg.update()
        await cl.Message(content=f" Fehler während der Indizierung: {str(e)}").send()


