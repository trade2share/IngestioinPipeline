# RAG-Ingestion Pipeline

Die Pipeline lädt PDF-Dokumente aus Azure Blob Storage, teilt sie in Chunks, bereinigt den Text und reichert die Metadaten an.
Abschließend werden Embeddings erzeugt und die Chunks in Azure AI Search (Vector Search) gespeichert. Klicke dafür einfach auf Starten und der Prozess wird getriggert.

