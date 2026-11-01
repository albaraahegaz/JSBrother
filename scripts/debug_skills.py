import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from jsb.skills import match_skill, get_minilm_matcher, normalize_skill, get_alias_registry
from jsb.agents.matcher import compute_match_score_and_verdict
from jsb.schemas import ProfileJSON, PostingJSON

with open("data/profile.json", encoding="utf-8") as f:
    prof_data = json.load(f)
profile = ProfileJSON(**prof_data)
skills = profile.skills

matcher = get_minilm_matcher()
print(f"MiniLM model loaded: {matcher.model is not None}")

# Check real_job_1 must haves
must_rj1 = [
    "machine learning",
    "neural networks",
    "pattern recognition",
    "computer science",
    "statistics",
    "applied mathematics",
    "natural language processing | nlp",
    "computer vision",
    "large language models | llm | llms",
    "pytorch | tensorflow | hugging face",
    "python",
    "pl/sql | plsql",
    "databases | database",
]

import numpy as np

print(f"Total candidate skills ({len(skills)}): {skills}\n")

def inspect_skill(req, candidate_skills):
    norm_req = normalize_skill(req)
    # Check if there are pipe alternatives in req!
    pipe_alts = [normalize_skill(p) for p in req.split("|")]
    
    cand_vecs = matcher.encode([normalize_skill(c) for c in candidate_skills])
    req_vec = matcher.encode([norm_req])[0]
    sims = np.dot(cand_vecs, req_vec)
    ranked = sorted(zip(candidate_skills, sims), key=lambda x: x[1], reverse=True)
    
    is_hit, matched, score, m_type = match_skill(req, candidate_skills)
    print(f"Skill: '{req}' -> is_hit={is_hit} (matched='{matched}', score={score:.3f}, type={m_type})")
    print(f"  Top 3 MiniLM matches: {[(c, round(float(s), 3)) for c, s in ranked[:3]]}")
    if len(pipe_alts) > 1:
        print(f"  Pipe alternatives: {pipe_alts}")
        for alt in pipe_alts:
            alt_vec = matcher.encode([alt])[0]
            alt_sims = np.dot(cand_vecs, alt_vec)
            alt_best = max(zip(candidate_skills, alt_sims), key=lambda x: x[1])
            print(f"    Alt '{alt}' best: {alt_best[0]} ({alt_best[1]:.3f})")

must_rp1 = [
    "python",
    "deep learning",
    "computer vision",
    "pytorch | tensorflow",
    "transfer learning",
    "cnns | convolutional neural networks",
    "data augmentation",
    "model evaluation",
    "jupyter notebook | jupyter",
]

print("=== REAL JOB 1 ===")
for r in must_rj1:
    inspect_skill(r, skills)

print("\n=== REAL PROJECT 1 ===")
for r in must_rp1:
    inspect_skill(r, skills)


