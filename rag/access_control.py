"""
Access Control.
Layer 6: RBAC access filter for retrieved document chunks.
Filters out documents the user's tier is not permitted to see.
"""

import logging
from typing import List, Dict
from config import TIER_PERMISSIONS

logger = logging.getLogger("cortex.access_control")


def filter_by_tier(chunks: List[Dict], user_tier: str) -> List[Dict]:
    """
    Filter retrieved chunks to only those whose access_tier is allowed for the user_tier.

    Args:
        chunks:    List of chunk dicts, each with an 'access_tier' key.
        user_tier: 'standard' | 'manager' | 'exec'

    If the document has no tag, the guard assumes it is "public".
    Is this tag on the user's allowed list?
        If Yes, the document is put into the new filtered list.
        If No, the document is thrown away.

    Returns:
        Filtered list:
        chunks with unauthorized access_tier removed.
    """
    allowed = TIER_PERMISSIONS.get(user_tier, {"public"})
    before = len(chunks)
    filtered = [c for c in chunks if c.get("access_tier", "public") in allowed]
    removed = before - len(filtered)

    if removed:
        logger.info(
            "[RBAC] Filtered %d chunk(s) for tier=%s (allowed=%s)",
            removed,
            user_tier,
            allowed,
        )

    return filtered
