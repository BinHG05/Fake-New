"""Strict Pydantic contracts for canonical fake-news pipeline records."""

from __future__ import annotations

from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ContractViolation(ValueError):
    """Raised when a source record cannot satisfy the canonical contract."""


class MediaContract(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    url: str = ""
    processed_path: str = ""
    available: bool = False


class CascadeCommentContract(BaseModel):
    """A normalized comment node in a propagation cascade."""

    model_config = ConfigDict(extra="allow", str_strip_whitespace=True)

    id: str = Field(min_length=1)
    parent_id: str = ""
    text: str = Field(min_length=1)
    user_id: Optional[str] = None
    timestamp: Optional[Union[int, float]] = None
    level: Optional[int] = Field(default=None, ge=1)


class CanonicalPostContract(BaseModel):
    """Schema persisted by the Bronze-to-Silver ETL boundary."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    id: str = Field(min_length=1)
    source: str = Field(min_length=1)
    raw_text: str = Field(min_length=1)
    clean_text: str = Field(min_length=1)
    label_binary: Literal["REAL", "FAKE"]
    split: Literal["train", "val", "test"]
    media: MediaContract
    cascade: list[CascadeCommentContract] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("source")
    @classmethod
    def normalize_source(cls, value: str) -> str:
        normalized = value.lower().strip()
        if not normalized:
            raise ValueError("source must not be blank")
        return normalized


__all__ = [
    "CanonicalPostContract",
    "CascadeCommentContract",
    "ContractViolation",
    "MediaContract",
]
