"""
Layer 5 — Query expansion, reformulation and intent classification.
Improves retrieval recall by broadening narrow or ambiguous queries.
"""

from numpy import true_divide
import logging
from typing import Dict
from config import LLM_MODEL, OPEN_API_KEY
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

logger = logging.getLogger("cortex.query_understanding")

# Fast, no-LLM mock expansions for common enterprise queries
MOCK_EXPANSIONS = {
    "pto": "paid time off annual leave vacation days holiday entitlement",
    "parental leave": "maternity leave paternity leave family leave parental policy",
    "wfh": "work from home remote work hybrid policy telecommute",
    "vpn": "virtual private network VPN setup remote access configuration",
    "onboarding": "new hire orientation first day checklist employee onboarding",
    "benefits": "employee benefits health insurance dental vision 401k pension",
    "performance": "performance review annual review feedback KPI goals",
    "expense": "expense reimbursement travel allowance business expense claim",
    "it help": "IT support helpdesk technical issue troubleshooting",
    "password": "password reset authentication login credentials access",
    "llm": "large language model artificial intelligence machine learning AI",
    "rag": "retrieval augmented generation document search vector database",
    "payroll": "salary payslip payment compensation wages",
    "offboarding": "resignation exit process last day equipment return",
}

_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
    return _llm


def understand_query(query: str) -> Dict[str, str]:
    """
    Reformulate and expand a query for improved retrieval.

    Returns dict with keys:
        original     — the raw input query
        expanded     — the enriched query (for embedding + BM25)
        intent       — coarse intent: 'lookup' | 'action' | 'research'
    """
    query_lower = query.lower()

    for key, expansion in MOCK_EXPANSIONS.items():
        if key in query_lower:
            logger.debug("[QueryUnderstanding] mock expansion matched: '%s'", key)
            return {
                "original": query,
                "expanded": f"{query} {expansion}",
                "intent": "lookup",
            }

    # LLM based expansion for queries not in mock dict
    api_key = OPEN_API_KEY
    if not api_key or api_key.startswith("sk-proj-"):
        try:
            logger.warning(
                "[QueryUnderstanding] No valid OpenAI API key, falling back to simple expansion"
            )
            llm = _get_llm()
            resp = llm.invoke(
                [
                    SystemMessage(
                        content=(
                            "You are a query expansion assistant for an enterprise knowledge base. "
                            "Given a user query, output an expanded version that includes "
                            "3-5 synonymous phrases, related terms, or alternative phrasings "
                            "to improve document retrieval. Return the expanded query only — "
                            "no explanation, no bullet points."
                        )
                    ),
                    HumanMessage(content=query),
                ]
            )
            expanded = resp.content.strip()
            logger.debug(
                "[QueryUnderstanding] LLM expanded: %r → %r", query[:60], expanded[:80]
            )
            return {"original": query, "expanded": expanded, "intent": "lookup"}
        except Exception as e:
            logger.warning("[QueryUnderstanding] LLM expansion failed: %s", e)
