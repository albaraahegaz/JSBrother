"""
JSBrother — FAISS Vector Store & Semantic Retriever Module
Builds and queries FAISS (IndexFlatL2 on normalized vectors) for:
1. Playbook rules (data/playbook/playbook.md) tagged by mode (job, freelance, both)
2. CV chunks from candidate ProfileJSON (roles, projects, skills, education)
Uses sentence-transformers/all-MiniLM-L6-v2 via skills.get_minilm_matcher().
"""

import re
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import numpy as np

import config
from jsb.schemas import ProfileJSON
from jsb.skills import get_minilm_matcher

logger = logging.getLogger("jsbrother.vector_store")


class FAISSIndex:
    """
    Wrapper around FAISS IndexFlatL2 on L2-normalized vectors.
    Provides fast cosine similarity retrieval over text chunks with metadata.
    """

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.texts: List[str] = []
        self.metadatas: List[Dict[str, Any]] = []
        self._index = None
        self._init_index()

    def _init_index(self):
        try:
            import faiss
            self._index = faiss.IndexFlatL2(self.dimension)
        except Exception as e:
            logger.warning(f"Could not initialize native FAISS ({e}), using numpy fallback.")
            self._index = None

    def add_texts(self, texts: List[str], embeddings: np.ndarray, metadatas: Optional[List[Dict[str, Any]]] = None):
        """Adds normalized embeddings and associated text/metadata to index."""
        if not texts or len(texts) == 0:
            return
        if metadatas is None:
            metadatas = [{} for _ in texts]

        # Ensure L2 normalized
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        norm_embs = (embeddings / norms).astype(np.float32)

        self.texts.extend(texts)
        self.metadatas.extend(metadatas)

        if self._index is not None:
            self._index.add(norm_embs)
        else:
            # Fallback storage in numpy array
            if not hasattr(self, "_raw_embs") or self._raw_embs is None:
                self._raw_embs = norm_embs
            else:
                self._raw_embs = np.vstack([self._raw_embs, norm_embs])

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 3,
        filter_fn: Optional[Any] = None,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Searches index for nearest neighbors.
        Returns list of (text, similarity_score, metadata).
        """
        if len(self.texts) == 0:
            return []

        # Normalize query vector
        norm = np.linalg.norm(query_embedding)
        norm_q = (query_embedding / (norm if norm > 0 else 1.0)).astype(np.float32)
        if norm_q.ndim == 1:
            norm_q = norm_q.reshape(1, -1)

        # Retrieve more candidates if filtering
        fetch_k = min(len(self.texts), top_k * 4 if filter_fn else top_k)

        results = []
        if self._index is not None:
            distances, indices = self._index.search(norm_q, fetch_k)
            # For normalized vectors, cosine_sim = 1 - (L2_dist^2)/2
            for dist, idx in zip(distances[0], indices[0]):
                if idx < 0 or idx >= len(self.texts):
                    continue
                sim = float(max(0.0, 1.0 - (dist / 2.0)))
                text = self.texts[idx]
                meta = self.metadatas[idx]
                if filter_fn and not filter_fn(meta):
                    continue
                results.append((text, sim, meta))
                if len(results) >= top_k:
                    break
        else:
            # Numpy dot-product search
            sims = np.dot(self._raw_embs, norm_q[0])
            ranked = np.argsort(sims)[::-1]
            for idx in ranked:
                meta = self.metadatas[idx]
                if filter_fn and not filter_fn(meta):
                    continue
                results.append((self.texts[idx], float(sims[idx]), meta))
                if len(results) >= top_k:
                    break

        return results[:top_k]


class PlaybookRetriever:
    """
    Parses data/playbook/playbook.md and retrieves top-k rules matching mode and query.
    """

    def __init__(self, playbook_path: Optional[Path] = None):
        self.playbook_path = playbook_path or (config.BASE_DIR / "data" / "playbook" / "playbook.md")
        self.matcher = get_minilm_matcher()
        self.index = FAISSIndex(dimension=384)
        self.rules: List[Dict[str, Any]] = []
        self._load_and_build()

    def _load_and_build(self):
        if not self.playbook_path.exists():
            logger.warning(f"Playbook file not found at {self.playbook_path}")
            return

        text = self.playbook_path.read_text(encoding="utf-8")
        parsed_rules = []
        # Match lines like: 1. [job] Rule text or - [freelance] Rule text
        pattern = re.compile(r"^\s*(?:\d+[\.\)]|\-)\s*\[(job|freelance|both)\]\s*(.+)$", re.IGNORECASE | re.MULTILINE)
        for match in pattern.finditer(text):
            mode_tag = match.group(1).lower()
            rule_content = match.group(2).strip()
            parsed_rules.append({
                "mode": mode_tag,
                "text": rule_content,
                "full": f"[{mode_tag}] {rule_content}",
            })

        self.rules = parsed_rules
        if not parsed_rules:
            return

        rule_texts = [r["text"] for r in parsed_rules]
        embs = self.matcher.encode(rule_texts)
        if embs is not None:
            self.index.add_texts(rule_texts, embs, metadatas=parsed_rules)

    def retrieve(self, query: str, mode: str = "both", top_k: int = 3) -> List[str]:
        """
        Retrieves top_k rules for the given query, filtered by mode.
        mode: 'job', 'project'/'freelance', or 'both'.
        """
        if not self.rules:
            return []

        active_mode = "freelance" if mode in ("freelance", "project") else ("job" if mode == "job" else "both")

        def mode_filter(meta: Dict[str, Any]) -> bool:
            tag = meta.get("mode", "both")
            if tag == "both":
                return True
            if active_mode == "both":
                return True
            return tag == active_mode

        q_emb = self.matcher.encode([query])
        if q_emb is None:
            # Fallback simple keyword match
            matches = [r["full"] for r in self.rules if mode_filter(r)]
            return matches[:top_k]

        search_res = self.index.search(q_emb[0], top_k=top_k, filter_fn=mode_filter)
        return [f"[{meta.get('mode', 'both')}] {text}" for text, _, meta in search_res]


class CVRetriever:
    """
    Chunks a candidate's ProfileJSON into semantic units and indexes them in FAISS.
    Retrieves the top-k most relevant CV chunks for a posting's requirements.
    """

    def __init__(self, profile: ProfileJSON):
        self.profile = profile
        self.matcher = get_minilm_matcher()
        self.index = FAISSIndex(dimension=384)
        self.chunks: List[Dict[str, Any]] = []
        self._build_index()

    def _build_index(self):
        chunks = []

        # 1. Profile Headline & Overview
        if self.profile.headline:
            chunks.append({
                "text": f"Candidate Overview: {self.profile.name}, {self.profile.headline}. Total experience: {self.profile.years_experience} years.",
                "section": "headline",
            })

        # 2. Roles & Highlights (chunked per role/highlight for fine-grained retrieval)
        for role in self.profile.roles:
            role_header = f"Experience at {role.company} as {role.title} ({role.years} yrs):"
            for hl in role.highlights:
                if hl.strip():
                    chunks.append({
                        "text": f"{role_header} {hl.strip()}",
                        "section": "experience",
                        "company": role.company,
                        "title": role.title,
                    })

        # 3. Projects
        for proj in self.profile.projects:
            if proj.strip():
                chunks.append({
                    "text": f"Project: {proj.strip()}",
                    "section": "project",
                })

        # 4. Core Skills
        if self.profile.skills:
            skills_str = ", ".join(self.profile.skills)
            chunks.append({
                "text": f"Verified Skills: {skills_str}",
                "section": "skills",
            })

        # 5. Education
        for edu in self.profile.education:
            if edu.strip():
                chunks.append({
                    "text": f"Education: {edu.strip()}",
                    "section": "education",
                })

        self.chunks = chunks
        if not chunks:
            return

        texts = [c["text"] for c in chunks]
        embs = self.matcher.encode(texts)
        if embs is not None:
            self.index.add_texts(texts, embs, metadatas=chunks)

    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        """
        Retrieves top_k CV chunks relevant to query (e.g., posting must-haves).
        """
        if not self.chunks:
            return []

        q_emb = self.matcher.encode([query])
        if q_emb is None:
            return [c["text"] for c in self.chunks[:top_k]]

        search_res = self.index.search(q_emb[0], top_k=top_k)
        return [text for text, _, _ in search_res]


def build_playbook_retriever() -> PlaybookRetriever:
    """Factory function for PlaybookRetriever."""
    return PlaybookRetriever()


def build_cv_retriever(profile: ProfileJSON) -> CVRetriever:
    """Factory function for CVRetriever."""
    return CVRetriever(profile)
