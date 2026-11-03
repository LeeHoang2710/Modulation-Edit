"""Render ViCo templates for subject-noun and modifier-only experiments."""
from __future__ import annotations

import re


def render_prompt(template: str, class_name: str, mode: str = "class") -> str:
    if template.count("{}") != 1:
        raise ValueError(f"Expected one placeholder in ViCo template: {template!r}")
    if mode == "class":
        return template.format(class_name)
    if mode == "modifier-only":
        # "a {} in the snow" -> "in the snow"; "a red {}" -> "red".
        prompt = re.sub(r"\s+", " ", template.replace("{}", " ")).strip()
        prompt = re.sub(r"^(?:a|an)\s+", "", prompt, flags=re.IGNORECASE).strip()
        if not prompt:
            raise ValueError(f"Template has no modifier after removing the object: {template!r}")
        return prompt
    raise ValueError(f"Unknown ViCo prompt mode: {mode}")
