import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from openai import OpenAI

load_dotenv()

# ── Config ───────────────────────────────────────────────────────────────────

EMBEDDING_MODEL = "nvidia/nemotron-3.5-embed-8b"
EMBEDDING_DIM = 4096
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

# ── NVIDIA Embeddings ────────────────────────────────────────────────────────

nvidia_client = OpenAI(
    base_url=NVIDIA_BASE_URL,
    api_key=os.environ["NVIDIA_API_KEY"],
)


class NVIDIAEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        response = nvidia_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=texts,
            extra_body={"input_type": "passage"},
        )
        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> list[float]:
        response = nvidia_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=[text],
            extra_body={"input_type": "query"},
        )
        return response.data[0].embedding


embeddings = NVIDIAEmbeddings()

# ── Qdrant ──────────────────────────────────────────────────────────────────

qdrant_client = QdrantClient(
    url=os.environ["QDRANT_URL"],
    api_key=os.environ["QDRANT_API_KEY"],
    timeout=120,
)


# ── Collection ───────────────────────────────────────────────────────────────

def get_collection_name(session_id: str) -> str:
    return f"papeer_nemotron_{session_id.replace('-', '_')}"


def get_vectorstore(session_id: str) -> QdrantVectorStore:
    collection_name = get_collection_name(session_id)

    if not qdrant_client.collection_exists(collection_name):
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=EMBEDDING_DIM,
                distance=Distance.COSINE,
            ),
        )

    return QdrantVectorStore(
        client=qdrant_client,
        collection_name=collection_name,
        embedding=embeddings,
    )


# ── Public API ───────────────────────────────────────────────────────────────

def add_paper(docs: list[Document], session_id: str) -> None:
    get_vectorstore(session_id).add_documents(docs)


def list_papers(session_id: str) -> list[str]:
    collection_name = get_collection_name(session_id)

    if not qdrant_client.collection_exists(collection_name):
        return []

    seen: set[str] = set()
    titles: list[str] = []
    offset = None

    while True:
        points, offset = qdrant_client.scroll(
            collection_name=collection_name,
            with_payload=True,
            limit=100,
            offset=offset,
        )

        for point in points:
            title = (point.payload or {}).get("metadata", {}).get("title")

            if title and title not in seen:
                seen.add(title)
                titles.append(title)

        if offset is None:
            break

    return titles


def search(query: str, session_id: str, k: int = 4) -> list[Document]:
    return get_vectorstore(session_id).similarity_search(query, k=k)