"""
JSBrother — Tests for Vector Store (FAISS) & Retrievers (M5)
Validates:
- Playbook parsing and mode filtering (job, freelance, both)
- Playbook FAISS indexing and top-k retrieval
- CV semantic chunking and FAISS retrieval
"""

import pytest
from jsb.schemas import ProfileJSON, RoleItem
from jsb.tools.vector_store import (
    FAISSIndex,
    PlaybookRetriever,
    CVRetriever,
    build_playbook_retriever,
    build_cv_retriever,
)


@pytest.fixture
def sample_profile():
    return ProfileJSON(
        name="Alex Smith",
        headline="AI & Machine Learning Engineer",
        years_experience=3.0,
        skills=["python", "pytorch", "fastapi", "rag", "docker"],
        roles=[
            RoleItem(
                title="AI Engineer",
                company="TechCorp",
                years=2.0,
                highlights=[
                    "Engineered an automated RAG workflow with vector search and reranking.",
                    "Optimized PyTorch inference latency by 35% using ONNX runtime.",
                ],
            )
        ],
        projects=[
            "Clinical Search: Built a biomedical semantic search engine using FastAPI and FAISS."
        ],
        education=["B.S. in Computer Science"],
    )


def test_playbook_retriever_rules_and_filtering():
    retriever = build_playbook_retriever()
    assert len(retriever.rules) > 0
    assert len(retriever.rules) <= 40

    # Job retrieval should not return [freelance] rules
    job_rules = retriever.retrieve("Python machine learning engineer", mode="job", top_k=3)
    assert len(job_rules) == 3
    for r in job_rules:
        assert not r.startswith("[freelance]")

    # Freelance retrieval should not return [job] rules
    fl_rules = retriever.retrieve("ImageNet classification notebook development", mode="freelance", top_k=3)
    assert len(fl_rules) == 3
    for r in fl_rules:
        assert not r.startswith("[job]")


def test_cv_retriever_chunking_and_retrieval(sample_profile):
    cv_retriever = build_cv_retriever(sample_profile)
    assert len(cv_retriever.chunks) >= 3

    # Query for RAG should retrieve the RAG role highlight or project
    results = cv_retriever.retrieve("Retrieval Augmented Generation and vector search", top_k=2)
    assert len(results) == 2
    combined = " ".join(results).lower()
    assert "rag" in combined or "vector" in combined or "search" in combined
