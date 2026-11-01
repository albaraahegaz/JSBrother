"""
JSBrother — Skills Normalization & Semantic Matching Module
Performs skill normalization, alias mapping ("a | b"), and MiniLM cosine similarity matching.
Adheres to §6:
1. Normalize (lowercase, trim, punctuation normalization)
2. Alias mapping (reuse convention "a | b" from my_cv_skills.txt and standard tech aliases)
3. MiniLM cosine similarity >= SKILL_SIM_THRESHOLD (from config.py)
-> Prevents false negatives when comparing technical skills using a 4-tier matching hierarchy.
"""

import os
import re
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Set, Optional, Any

import config

logger = logging.getLogger("jsbrother.skills")

# Built-in standard aliases for tech, libraries, and frameworks
DEFAULT_ALIASES = [
    # Programming languages & core concepts
    "python | py",
    "javascript | js",
    "typescript | ts",
    "golang | go",
    "c++ | cpp",
    "c# | csharp | c-sharp",
    "object-oriented programming | oop",
    "object oriented programming | oop",
    "functional programming | fp",
    "data structures and algorithms | dsa",
    
    # ML / AI / Data
    "machine learning | ml | pattern recognition",
    "deep learning | dl | neural networks | artificial neural networks | ann | nns",
    "artificial intelligence | ai",
    "natural language processing | nlp",
    "large language models | llm | llms | generative ai | genai",
    "agentic ai | agentic systems | ai agents | autonomous agents",
    "retrieval-augmented generation | rag | retrieval augmented generation",
    "exploratory data analysis | eda",
    "computer vision | cv | image processing | image classification | object detection | ocr",
    "transfer learning | transfer-learning | pretrained models | fine-tuning | finetuning | transformers",
    "convolutional neural networks | cnns | cnn | convnets",
    "recurrent neural networks | rnns | rnn",
    "reinforcement learning | rl",
    "scikit-learn | sklearn",
    "pytorch | torch",
    "tensorflow | tf",
    "keras | tf.keras",
    "transformers | huggingface | hugging face",
    "langchain | lc",
    "langgraph | lg",
    "microsoft excel | excel",
    "power bi | powerbi",
    "jupyter notebook | jupyter | ipython",
    "data cleaning | data cleansing | data preprocessing | data wrangling",
    "data visualization | data visualisation | dataviz | plotting | dashboards",
    "data augmentation | image augmentation | feature engineering",
    "model evaluation | model validation | evaluation metrics | validation metrics | evaluation | ragas | langsmith",
    "applied mathematics | mathematics | math | statistics | statistical analysis",
    "computer science | cs | software engineering | ai engineering",
    
    # Databases & Backend
    "database | databases | db | sql | relational database | rdbms",
    "pl/sql | plsql | t-sql | tsql | sql",
    "postgresql | postgres | pg",
    "sql server | mssql | ms sql",
    "mysql | my-sql",
    "sqlite | sqlite3",
    "mongodb | mongo",
    "redis | redis cache",
    "rest | rest api | rest apis | restful api | restful apis",
    "fastapi | fast api",
    "asyncio | async python | asynchronous programming",
    "node.js | nodejs | node",
    "next.js | nextjs | next",
    "react | react.js | reactjs",
    "vue | vue.js | vuejs",
    "tailwind css | tailwind | tailwindcss",
    "html5 | html",
    "css3 | css",
    
    # DevOps / Cloud / Tools
    "amazon web services | aws",
    "google cloud platform | google cloud | gcp",
    "microsoft azure | azure",
    "docker | containerization | containers",
    "kubernetes | k8s",
    "ci/cd | cicd | continuous integration | continuous deployment",
    "git | github | gitlab",
    "automated testing | testing | unit testing | integration testing",
    "tool use | function calling",
    "claude code | claude-code",
    "prompt engineering | prompting",
    "re-ranking | reranking",
    "evaluation | evaluation harnesses",
]


def normalize_skill(skill: str) -> str:
    """
    Normalizes a skill string:
    - lowercase
    - strip extra whitespace
    - canonicalize common punctuation patterns (e.g. next.js -> nextjs if needed, or stripped)
    """
    if not skill:
        return ""
    s = skill.strip().lower()
    # Normalize multiple whitespace
    s = re.sub(r"\s+", " ", s)
    # Strip leading/trailing non-alphanumeric except + / # (e.g. c++, c#)
    s = re.sub(r"^[^a-z0-9+#]+|[^a-z0-9+#]+$", "", s)
    return s


class SkillAliasRegistry:
    """
    Manages skill aliases using the "a | b" convention from my_cv_skills.txt,
    allowing bi-directional lookup of equivalent skills.
    """

    def __init__(self):
        self.alias_to_canonical: Dict[str, str] = {}
        self.canonical_to_group: Dict[str, Set[str]] = {}
        self._load_builtins()
        self._load_from_samples()

    def _load_builtins(self):
        for line in DEFAULT_ALIASES:
            self.register_alias_line(line)

    def _load_from_samples(self):
        # Look for my_cv_skills.txt
        sample_paths = [
            Path(config.BASE_DIR) / "data" / "samples" / "cvs" / "my_cv_skills.txt",
            Path(config.BASE_DIR) / "data" / "samples" / "postings" / "real_labels.txt",
        ]
        for path in sample_paths:
            if path.exists():
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line or line.startswith("#") or line.startswith("["):
                                continue
                            if ":" in line and not line.startswith("must_have"):
                                # Skip metadata lines like "type: job"
                                continue
                            if "|" in line:
                                self.register_alias_line(line)
                except Exception as e:
                    logger.debug(f"Could not load aliases from {path}: {e}")

    def register_alias_line(self, line: str):
        """
        Parses a line like 'scikit-learn | sklearn' or 'a | b | c'
        and maps all variants into the same alias group.
        """
        parts = [normalize_skill(p) for p in line.split("|") if normalize_skill(p)]
        if not parts:
            return
        canonical = parts[0]
        # If any part already has a canonical, merge groups
        for p in parts:
            if p in self.alias_to_canonical:
                canonical = self.alias_to_canonical[p]
                break

        group = self.canonical_to_group.setdefault(canonical, set())
        for p in parts:
            group.add(p)
            self.alias_to_canonical[p] = canonical

    def get_canonical(self, skill: str) -> str:
        """Returns the canonical skill name if known, else the normalized skill."""
        norm = normalize_skill(skill)
        return self.alias_to_canonical.get(norm, norm)

    def are_aliases(self, skill_a: str, skill_b: str) -> bool:
        """Checks if two skill strings are equivalent via exact or alias matching."""
        norm_a = normalize_skill(skill_a)
        norm_b = normalize_skill(skill_b)
        if not norm_a or not norm_b:
            return False
        if norm_a == norm_b:
            return True

        # Check canonical map
        can_a = self.alias_to_canonical.get(norm_a)
        can_b = self.alias_to_canonical.get(norm_b)
        if can_a and can_b and can_a == can_b:
            return True

        # Normalized without hyphens/dots comparison
        clean_a = re.sub(r"[\s\-_.]", "", norm_a)
        clean_b = re.sub(r"[\s\-_.]", "", norm_b)
        if clean_a == clean_b:
            return True

        return False


# Singleton registry
_registry: Optional[SkillAliasRegistry] = None


def get_alias_registry() -> SkillAliasRegistry:
    global _registry
    if _registry is None:
        _registry = SkillAliasRegistry()
    return _registry


# -----------------------------------------------------------------------------
# MiniLM Similarity Matcher
# -----------------------------------------------------------------------------

class MiniLMMatcher:
    """
    Computes embedding-based semantic similarity between skills using MiniLM.
    Lazily loads the SentenceTransformer model on first usage.
    """

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or getattr(
            config, "EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
        )
        self._model = None
        self._load_attempted = False

    @property
    def model(self):
        if self._model is None and not self._load_attempted:
            self._load_attempted = True
            try:
                from sentence_transformers import SentenceTransformer
                logger.info(f"Loading embedding model: {self.model_name}...")
                self._model = SentenceTransformer(self.model_name)
            except Exception as e:
                logger.warning(
                    f"SentenceTransformer could not be loaded ({e}). "
                    "Falling back to alias/token similarity."
                )
                self._model = None
        return self._model

    def encode(self, texts: List[str]):
        """Encodes a list of texts into normalized numpy vectors."""
        if not self.model or not texts:
            return None
        return self.model.encode(texts, normalize_embeddings=True)

    def similarity(self, text_a: str, text_b: str) -> float:
        """Computes cosine similarity between two single skill strings."""
        if not text_a or not text_b:
            return 0.0
        if self.model:
            import numpy as np
            emb = self.model.encode([text_a, text_b], normalize_embeddings=True)
            return float(np.dot(emb[0], emb[1]))
        return self._token_fallback_similarity(text_a, text_b)

    def _token_fallback_similarity(self, a: str, b: str) -> float:
        """Token-level overlap fallback when transformer model is not loaded."""
        set_a = set(re.findall(r"\w+", normalize_skill(a)))
        set_b = set(re.findall(r"\w+", normalize_skill(b)))
        if not set_a or not set_b:
            return 0.0
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        jaccard = intersection / union if union else 0.0
        if a in b or b in a:
            return max(jaccard, 0.75)
        return jaccard


_minilm_matcher: Optional[MiniLMMatcher] = None


def get_minilm_matcher() -> MiniLMMatcher:
    global _minilm_matcher
    if _minilm_matcher is None:
        _minilm_matcher = MiniLMMatcher()
    return _minilm_matcher


# -----------------------------------------------------------------------------
# High-Level Matching Functions
# -----------------------------------------------------------------------------

def match_skill(
    required_skill: str,
    candidate_skills: List[str],
    threshold: Optional[float] = None,
) -> Tuple[bool, Optional[str], float, str]:
    """
    Checks if a single required_skill matches any skill in candidate_skills.
    Order per §6:
    1. Exact / normalized match
    2. Alias map match ("a | b")
    3. MiniLM cosine similarity >= threshold (default 0.70 from config)

    Returns:
        (is_match, matched_candidate_skill, score, match_type)
        match_type: "exact", "alias", "semantic", or "none"
    """
    if threshold is None:
        threshold = getattr(config, "SKILL_SIM_THRESHOLD", 0.70)

    # If the required skill lists alternatives ("a | b | c"), evaluate each alternative
    if "|" in required_skill:
        alts = [p.strip() for p in required_skill.split("|") if p.strip()]
        if len(alts) > 1:
            best_res = (False, None, 0.0, "none")
            for alt in alts:
                res = match_skill(alt, candidate_skills, threshold=threshold)
                if res[0]:
                    if res[3] in ("exact", "alias"):
                        return res
                    if res[2] > best_res[2]:
                        best_res = res
                else:
                    if res[2] > best_res[2]:
                        best_res = res
            return best_res

    norm_req = normalize_skill(required_skill)
    if not norm_req:
        return False, None, 0.0, "none"

    registry = get_alias_registry()

    # Step 1 & 2: Exact and alias matching
    for cand in candidate_skills:
        norm_cand = normalize_skill(cand)
        if norm_req == norm_cand:
            return True, cand, 1.0, "exact"
        if registry.are_aliases(norm_req, norm_cand):
            return True, cand, 1.0, "alias"

    # Step 3: MiniLM semantic similarity
    matcher = get_minilm_matcher()
    if matcher.model is not None and candidate_skills:
        try:
            import numpy as np
            req_vec = matcher.encode([norm_req])[0]
            cand_vecs = matcher.encode([normalize_skill(c) for c in candidate_skills])
            sims = np.dot(cand_vecs, req_vec)
            best_idx = int(np.argmax(sims))
            best_score = float(sims[best_idx])
            if best_score >= threshold:
                return True, candidate_skills[best_idx], best_score, "semantic"
            return False, candidate_skills[best_idx], best_score, "none"
        except Exception as e:
            logger.debug(f"MiniLM matrix similarity error: {e}")

    # Fallback single-pair semantic check
    best_cand = None
    best_score = 0.0
    for cand in candidate_skills:
        score = matcher.similarity(norm_req, normalize_skill(cand))
        if score > best_score:
            best_score = score
            best_cand = cand

    if best_score >= threshold:
        return True, best_cand, best_score, "semantic"

    return False, best_cand, best_score, "none"


def match_skills(
    required_skills: List[str],
    candidate_skills: List[str],
    threshold: Optional[float] = None,
) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
    """
    Evaluates required_skills against candidate_skills.

    Returns:
        (matched_skills, missing_skills, details)
        - matched_skills: subset of required_skills that were found
        - missing_skills: subset of required_skills not found
        - details: list of match info dicts for explainability
    """
    matched = []
    missing = []
    details = []

    for req in required_skills:
        is_hit, matched_cand, score, m_type = match_skill(
            req, candidate_skills, threshold=threshold
        )
        details.append({
            "required": req,
            "matched_candidate": matched_cand,
            "score": round(score, 3),
            "match_type": m_type,
            "matched": is_hit,
        })
        if is_hit:
            matched.append(req)
        else:
            missing.append(req)

    return matched, missing, details


def extract_all_candidate_skills(profile) -> List[str]:
    """
    Combines profile.skills with technical skills explicitly present in roles,
    projects, and education text.
    """
    skills_set = {normalize_skill(s) for s in profile.skills if s}

    # Text corpus from experience, roles, projects, headline, education
    corpus_parts = [profile.headline or ""]
    for r in getattr(profile, "roles", []):
        corpus_parts.extend([getattr(r, "title", ""), getattr(r, "company", "")] + getattr(r, "highlights", []))
    corpus_parts.extend(getattr(profile, "projects", []))
    corpus_parts.extend(getattr(profile, "education", []))
    corpus = " ".join(corpus_parts).lower()

    # Check known aliases and vocabulary from alias registry
    registry = get_alias_registry()
    all_vocab = set(registry.alias_to_canonical.keys()) | set(registry.canonical_to_group.keys())

    for term in all_vocab:
        if len(term) >= 3 and re.search(r"\b" + re.escape(term) + r"\b", corpus):
            skills_set.add(term)

    result = list(profile.skills)
    existing_norm = {normalize_skill(x) for x in result}
    for s in sorted(skills_set):
        norm_s = normalize_skill(s)
        if norm_s not in existing_norm:
            result.append(s)
            existing_norm.add(norm_s)
    return result

