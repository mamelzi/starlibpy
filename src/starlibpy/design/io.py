"""Serialization helpers for starlibpy design objects."""

from __future__ import annotations

from pathlib import Path
from typing import Literal
import json

from .models import AnalysisDesign, StudyDesign


def export_design(
    design: StudyDesign | AnalysisDesign,
    path: str | Path,
    *,
    format: Literal["json", "yaml", "yml"] | None = None,
) -> Path:
    """Export a StudyDesign or AnalysisDesign to JSON or YAML atomically."""
    if not isinstance(design, (StudyDesign, AnalysisDesign)):
        raise TypeError("design must be a StudyDesign or AnalysisDesign.")

    output = Path(path)
    selected = (format or output.suffix.lstrip(".") or "json").lower()
    if selected == "yml":
        selected = "yaml"
    if selected not in {"json", "yaml"}:
        raise ValueError("format must be 'json' or 'yaml'.")
    if not output.suffix:
        output = output.with_suffix(".yaml" if selected == "yaml" else ".json")
    output.parent.mkdir(parents=True, exist_ok=True)

    payload = design.to_dict()
    temporary = output.with_suffix(output.suffix + ".tmp")
    if selected == "json":
        text = json.dumps(payload, ensure_ascii=False, indent=2)
    else:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover
            raise ImportError("PyYAML is required for YAML export.") from exc
        text = yaml.safe_dump(payload, allow_unicode=True, sort_keys=False)
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(output)
    return output


def load_design(path: str | Path) -> StudyDesign | AnalysisDesign:
    """Load a serialized StudyDesign or AnalysisDesign."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)

    if source.suffix.lower() == ".json":
        payload = json.loads(source.read_text(encoding="utf-8"))
    elif source.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover
            raise ImportError("PyYAML is required for YAML import.") from exc
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    else:
        raise ValueError("File extension must be .json, .yaml, or .yml.")

    object_type = payload.get("object_type")
    if object_type == "StudyDesign":
        return StudyDesign.from_dict(payload)
    if object_type == "AnalysisDesign":
        return AnalysisDesign.from_dict(payload)
    raise ValueError(f"Unsupported serialized object_type: {object_type!r}.")
