"""
JSBrother — Data Schemas
Pydantic v2 data models for Profile, Posting, MatchResult, and Draft.
Defines strict schemas with forgiving defaults for LLM extraction per §5.
"""

from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field, ConfigDict, field_validator


class RoleItem(BaseModel):
    """Past work experience entry in a candidate profile."""
    model_config = ConfigDict(extra="ignore")

    title: str = ""
    company: str = ""
    years: float = 0.0
    highlights: List[str] = Field(default_factory=list)


class BudgetInfo(BaseModel):
    """Budget specification for freelance projects."""
    model_config = ConfigDict(extra="ignore")

    amount: float = 0.0
    currency: str = "USD"
    kind: Literal["fixed", "hourly"] = "fixed"


class ProfileJSON(BaseModel):
    """Structured candidate or freelancer profile extracted from CV/text."""
    model_config = ConfigDict(extra="ignore")

    name: str = ""
    email: str = ""
    headline: str = ""
    skills: List[str] = Field(default_factory=list)
    years_experience: float = 0.0
    roles: List[RoleItem] = Field(default_factory=list)
    education: List[str] = Field(default_factory=list)
    projects: List[str] = Field(default_factory=list)
    languages: List[str] = Field(default_factory=list)
    links: List[str] = Field(default_factory=list)
    min_budget: Optional[float] = None
    min_lead_days: int = 7

    @field_validator("name", "email", "headline", mode="before")
    @classmethod
    def _coerce_profile_str(cls, v: Any) -> str:
        return "" if v is None else str(v)

    @field_validator("skills", "education", "projects", "languages", "links", mode="before")
    @classmethod
    def _coerce_profile_list(cls, v: Any) -> list:
        if v is None:
            return []
        if isinstance(v, str):
            return [v]
        return list(v)

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class PostingJSON(BaseModel):
    """Structured job or freelance project posting details."""
    model_config = ConfigDict(extra="ignore")

    title: str = ""
    company_or_client: str = ""
    type: Literal["job", "project"] = "job"
    must_have: List[str] = Field(default_factory=list)
    nice_to_have: List[str] = Field(default_factory=list)
    min_years: Optional[float] = None
    deal_breakers: List[str] = Field(default_factory=list)
    budget: Optional[BudgetInfo] = None
    deadline_days: Optional[int] = None
    vague_scope: bool = False
    summary: str = ""
    contact_email: Optional[str] = None

    @field_validator("title", "company_or_client", "summary", mode="before")
    @classmethod
    def _coerce_posting_str(cls, v: Any) -> str:
        return "" if v is None else str(v)

    @field_validator("must_have", "nice_to_have", "deal_breakers", mode="before")
    @classmethod
    def _coerce_posting_list(cls, v: Any) -> list:
        if v is None:
            return []
        if isinstance(v, str):
            return [v]
        return list(v)

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class FreelancerMatch(BaseModel):
    """Freelancer-specific matching details."""
    model_config = ConfigDict(extra="ignore")

    budget_fit: float = 1.0
    deadline_fit: float = 1.0
    red_flags: List[str] = Field(default_factory=list)


class MatchResult(BaseModel):
    """Deterministic match evaluation and verdict computed by Python."""
    model_config = ConfigDict(extra="ignore")

    score: float = Field(ge=0.0, le=100.0, default=0.0)
    must_coverage: float = Field(ge=0.0, le=1.0, default=0.0)
    nice_coverage: float = Field(ge=0.0, le=1.0, default=0.0)
    matched: List[str] = Field(default_factory=list)
    missing_must: List[str] = Field(default_factory=list)
    missing_nice: List[str] = Field(default_factory=list)
    deal_breaker_hits: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    verdict: Literal["APPLY", "CAUTION", "SKIP"] = "CAUTION"
    reasons: List[str] = Field(default_factory=list)
    advice: str = ""
    freelancer: Optional[FreelancerMatch] = None

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class Draft(BaseModel):
    """Generated application draft (cover letter or proposal) with claim audit."""
    model_config = ConfigDict(extra="ignore")

    kind: Literal["cover_letter", "proposal", "email_message", "short_bid"] = "cover_letter"
    subject: str = ""
    body: str = ""
    claims_ok: bool = True
    flagged_claims: List[str] = Field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()
