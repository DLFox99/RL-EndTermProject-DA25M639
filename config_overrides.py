"""Typed dotted-path configuration overrides for experiments and sweeps."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Iterable, Tuple

import yaml


def parse_override(text: str) -> Tuple[str, Any]:
    if "=" not in text:
        raise ValueError(f"override must be KEY=VALUE, got: {text!r}")
    key, raw = text.split("=", 1)
    key = key.strip()
    if not key:
        raise ValueError("override key cannot be empty")
    value = yaml.safe_load(raw)
    # yaml.safe_load can fail to recognize scientific notation like
    # "3e-05" as a float (some PyYAML versions require a decimal point,
    # e.g. "3.0e-05"), silently returning it as a string instead. Retry
    # as float explicitly when yaml gives back a str that looks numeric.
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError:
            pass
    return key, value

#def parse_override(text: str) -> Tuple[str, Any]:
#    if "=" not in text:
#        raise ValueError(f"override must be KEY=VALUE, got: {text!r}")
#    key, raw = text.split("=", 1)
#    key = key.strip()
#    if not key:
#        raise ValueError("override key cannot be empty")
#    value = yaml.safe_load(raw)
#    return key, value


def _set_dotted(root: Dict[str, Any], dotted: str, value: Any) -> None:
    parts = [p for p in dotted.split(".") if p]
    if not parts:
        raise ValueError("override path cannot be empty")
    node = root
    for part in parts[:-1]:
        current = node.get(part)
        if current is None:
            current = {}
            node[part] = current
        if not isinstance(current, dict):
            raise ValueError(
                f"cannot descend through non-mapping path component {part!r} "
                f"in {dotted!r}"
            )
        node = current
    node[parts[-1]] = value


def apply_overrides(
    config: Dict[str, Any],
    overrides: Iterable[str],
    *,
    technique: str | None,
) -> Dict[str, Any]:
    """Return a deep-copied config with CLI overrides applied.

    For a single technique, keys that do not begin with an existing top-level
    config section are relative to ``techniques.<technique>``.  Therefore both
    ``epsilon_end=0.02`` and ``schedules.epsilon.type=cosine`` are convenient
    technique-local overrides, while ``evaluation.plateau.patience=6`` and
    ``techniques.nn_sarsa.epsilon_end=0.02`` are absolute.  With
    ``technique='all'``, only absolute paths are accepted.
    """
    out = deepcopy(config)
    for text in overrides:
        key, value = parse_override(text)
        first = key.split(".", 1)[0]
        is_absolute = first in out
        if not is_absolute:
            if technique in (None, "all"):
                raise ValueError(
                    f"relative override {key!r} requires one technique; use a full "
                    f"path such as techniques.nn_sarsa.{key}"
                )
            key = f"techniques.{technique}.{key}"
        _set_dotted(out, key, value)
    return out
