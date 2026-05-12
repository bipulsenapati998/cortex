"""
Unified loader for all versioned YAML prompts.

Each prompt YAML must contain these sections:
  - role_and_constraints   : Who the agent is + hard rules
  - context_and_examples   : Few-shot examples + format instructions
  - top_guard              : Security preamble (injected at top of system prompt)
  - bottom_guard           : Final self-check (injected at bottom)
  - system_prompt          : Pre-assembled full prompt (used directly by LLM calls)

The assembled system_prompt is the canonical field used at runtime.
The individual sections exist for documentation, auditing, and A/B testing.

Usage:
    from cortex.prompts.prompt_loader import load_prompt, load_prompt_sections

    # Simple — just get the system prompt string
    system_prompt = load_prompt("knowledge_agent", version="v1.0.0")

    # Advanced — get individual sections for inspection or custom assembly
    sections = load_prompt_sections("supervisor", version="v1.0.0")
    print(sections["top_guard"])
    print(sections["role_and_constraints"])
"""

import logging
from pathlib import Path
from typing import Dict, Optional
import yaml

logger = logging.getLogger("cortex.prompt_loader")

# Canonical prompt directory (relative to this file's location)
_PROMPTS_DIR = Path(__file__).parent

# Required sections in every prompt YAML
REQUIRED_SECTIONS = {
    "version",
    "agent",
    "role_and_constraints",
    "context_and_examples",
    "top_guard",
    "bottom_guard",
    "system_prompt",
}

# Fallback prompts when YAML file is missing or malformed
_FALLBACKS: Dict[str, str] = {
    "supervisor": (
        "Classify the user query as: knowledge, research, or action. "
        "Respond with exactly: 'intent: <knowledge|research|action>'"
    ),
    "knowledge_agent": (
        "You are CORTEX's Knowledge Agent. Answer ONLY from the provided context documents. "
        "Cite sources as [Doc N]. If the context lacks the answer, say so honestly."
    ),
    "research_agent": (
        "You are CORTEX's Research Agent. Summarise the search results clearly. "
        "Attribute all information as 'Based on web search results...'"
    ),
}


def load_prompt(
    agent: str,
    version: str = "v1.0.0",
    section: str = "system_prompt",
) -> str:
    """
    Load the assembled system prompt for the given agent and version.

    Args:
        agent:   Agent name — matches subdirectory under prompts/
                 e.g. "supervisor" → prompts/supervisor/v1.0.0.yaml
                 e.g. "knowledge_agent" → prompts/agents/knowledge_agent/v1.0.0.yaml
        version: Prompt version string, e.g. "v1.0.0", "v1.1.0"
        section: Which field to return (default: "system_prompt").
                 Can also be: "top_guard", "bottom_guard",
                              "role_and_constraints", "context_and_examples"

    Returns:
        The requested section text as a string.
        Falls back to a hardcoded default if the file is missing or malformed.
    """
    path = _resolve_path(agent, version)
    if path is None or not path.exists():
        logger.warning(
            "[PromptLoader] Prompt file not found for agent=%s version=%s — using fallback",
            agent, version,
        )
        return _FALLBACKS.get(agent, f"You are CORTEX's {agent}.")

    try:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except Exception as e:
        logger.error("[PromptLoader] YAML parse error for %s: %s — using fallback", path, e)
        return _FALLBACKS.get(agent, f"You are CORTEX's {agent}.")

    # Validate required sections
    missing = REQUIRED_SECTIONS - set(data.keys())
    if missing:
        logger.warning(
            "[PromptLoader] Prompt %s/%s is missing sections: %s — using system_prompt if present",
            agent, version, missing,
        )

    content = data.get(section, "")
    if not content:
        # Graceful degradation: fall through to system_prompt if requested section is empty
        content = data.get("system_prompt", _FALLBACKS.get(agent, ""))
        logger.warning(
            "[PromptLoader] Section '%s' empty in %s/%s — using system_prompt",
            section, agent, version,
        )

    logger.debug("[PromptLoader] Loaded %s/%s [section=%s] (%d chars)", agent, version, section, len(content))
    return content.strip()


def _resolve_path(agent: str, version: str) -> Optional[Path]:
    """
    Resolve YAML path.
    Searches: prompts/{agent}/{version}.yaml
              prompts/agents/{agent}/{version}.yaml
    """
    candidates = [
        _PROMPTS_DIR / agent / f"{version}.yaml",
        _PROMPTS_DIR / "agents" / agent / f"{version}.yaml",
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]  # return primary path even if missing (for logging)
