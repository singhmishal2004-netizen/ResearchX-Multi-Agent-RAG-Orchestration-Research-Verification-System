# ResearchX-Multi-Agent-RAG-Orchestration-Research-Verification-System

ResearchX uses NVIDIA's hosted `nvidia/nemotron-3.5-super-vl-120b-a12b` model
for chat and `nvidia/nemotron-3.5-embed-8b` for embeddings through the NVIDIA
OpenAI-compatible API. These models are API-served; they are not downloaded
into the local Python environment.

Set `NVIDIA_API_KEY`, `QDRANT_URL`, `QDRANT_API_KEY`, and `TAVILY_API_KEY` in
`.env`. Install the Python dependencies with
`python -m pip install -r requirements.txt`, then start the app with
`streamlit run app.py`.

Embeddings are 4,096 dimensions and are stored in per-session Qdrant
collections named `papeer_nemotron_<session id>`. Documents indexed with a
different embedding model must be uploaded again to create vectors compatible
with Nemotron Embed. Web search and claim verification use Tavily.