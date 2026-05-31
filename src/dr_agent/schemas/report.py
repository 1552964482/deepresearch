"""Report data models."""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field


class Citation(BaseModel):
    id: str
    title: str | None = None
    url: str | None = None
    snippet: str | None = None


class Section(BaseModel):
    heading: str
    body: str
    subtask_id: str | None = None
    citations: list[Citation] = Field(default_factory=list)


class ResearchReport(BaseModel):
    task_id: str
    user_query: str
    title: str
    summary: str
    sections: list[Section] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    degraded: bool = False
    degrade_reason: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def to_markdown(self) -> str:
        parts: list[str] = []
        parts.append(f"# {self.title}\n")
        parts.append(f"> Query: {self.user_query}")
        parts.append(f"> Generated: {self.created_at.isoformat()}")
        if self.degraded:
            parts.append(f"> ⚠️  Degraded: {self.degrade_reason}")
        parts.append("\n## Summary\n")
        parts.append(self.summary)
        for sec in self.sections:
            parts.append(f"\n## {sec.heading}\n")
            parts.append(sec.body)
            if sec.citations:
                parts.append("\n**References (this section):**\n")
                for c in sec.citations:
                    label = c.title or c.id
                    if c.url:
                        parts.append(f"- [{label}]({c.url})")
                    else:
                        parts.append(f"- {label}")
        if self.citations:
            parts.append("\n## All References\n")
            for c in self.citations:
                label = c.title or c.id
                if c.url:
                    parts.append(f"- [{label}]({c.url}) — {c.snippet or ''}")
                else:
                    parts.append(f"- {label} — {c.snippet or ''}")
        return "\n".join(parts) + "\n"
