"""Thin wrapper around the Anthropic Messages API that returns validated Pydantic objects."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Protocol, TypeVar

import anthropic
import httpx
from pydantic import BaseModel

from biorag.config import Settings

T = TypeVar("T", bound=BaseModel)


class LLM(Protocol):
    def structured(self, system: str, prompt: str, schema: type[T], effort: str = "low") -> T: ...


class RefusalError(RuntimeError):
    pass


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0


@dataclass
class ClaudeLLM:
    """Calls Claude with a JSON-schema output constraint.

    Server-side refusal fallbacks are on: biomedical questions occasionally trip safety
    classifiers, and the "default" fallback re-runs those requests on a suitable model
    within the same call.
    """

    model: str
    max_tokens: int = 8000
    client: anthropic.Anthropic = field(default_factory=anthropic.Anthropic)
    usage: Usage = field(default_factory=Usage)

    def structured(self, system: str, prompt: str, schema: type[T], effort: str = "low") -> T:
        response = self.client.beta.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            output_config={
                "effort": effort,
                "format": {"type": "json_schema", "schema": _strict_schema(schema)},
            },
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        self.usage.calls += 1
        self.usage.input_tokens += response.usage.input_tokens
        self.usage.output_tokens += response.usage.output_tokens

        if response.stop_reason == "refusal":
            raise RefusalError(f"{self.model} declined the request: {response.stop_details}")
        if response.stop_reason == "max_tokens":
            raise RuntimeError(f"Response hit max_tokens={self.max_tokens}")
        text = next(b.text for b in response.content if b.type == "text")
        return schema.model_validate(json.loads(text))


@dataclass
class OllamaLLM:
    """Free local backend via Ollama (https://ollama.com). Uses Ollama's JSON-schema `format`
    to constrain output. `effort` has no equivalent and is ignored."""

    model: str
    base_url: str = "http://localhost:11434"
    timeout: float = 600.0
    usage: Usage = field(default_factory=Usage)

    def structured(self, system: str, prompt: str, schema: type[T], effort: str = "low") -> T:
        r = httpx.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                "format": _strict_schema(schema),
                "stream": False,
                "options": {"temperature": 0, "num_ctx": 8192},
            },
            timeout=self.timeout,
        )
        r.raise_for_status()
        data = r.json()
        self.usage.calls += 1
        self.usage.input_tokens += data.get("prompt_eval_count", 0)
        self.usage.output_tokens += data.get("eval_count", 0)
        return schema.model_validate(json.loads(data["message"]["content"]))


def make_llm(settings: Settings, role: str = "generator") -> LLM:
    """Build the generator or judge model for the configured backend."""
    if settings.llm_backend == "ollama":
        model = settings.ollama_judge_model or settings.ollama_model
        return OllamaLLM(model if role == "judge" else settings.ollama_model, settings.ollama_url)
    if settings.llm_backend == "claude":
        return ClaudeLLM(settings.judge_model if role == "judge" else settings.generator_model)
    raise ValueError(f"unknown llm_backend: {settings.llm_backend}")


def _strict_schema(model: type[BaseModel]) -> dict:
    """Pydantic JSON schema with additionalProperties=false on every object (API requirement)."""
    schema = model.model_json_schema()

    def visit(node: object) -> None:
        if isinstance(node, dict):
            if node.get("type") == "object":
                node["additionalProperties"] = False
                node.setdefault("required", list(node.get("properties", {})))
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for item in node:
                visit(item)

    visit(schema)
    return schema
