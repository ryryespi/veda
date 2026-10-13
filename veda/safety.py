"""Crisis gate. Veda is a reflection aid, not therapy. Any crisis language
suspends scoring and returns only support resources (US: 988)."""
import re

CRISIS_MESSAGE = (
    "You deserve support right now. Veda is not therapy and cannot help in a crisis. "
    "In the US, call or text 988 (Suicide & Crisis Lifeline), or call 911 if you are in "
    "immediate danger. Outside the US, contact your local emergency number."
)

_PATTERNS = [
    r"\bkill(ing)?\s+my\s*self\b", r"\bsuicid", r"\bend\s+(my|it)\s+(life|all)\b",
    r"\bwant\s+to\s+die\b", r"\bself[-\s]?harm", r"\bhurt\s+my\s*self\b",
    r"\bno\s+reason\s+to\s+live\b", r"\bbetter\s+off\s+dead\b",
]
_RX = re.compile("|".join(_PATTERNS), re.IGNORECASE)


def crisis_check(text: str | None) -> bool:
    """True if the text contains crisis language. Keyword matching is crude:
    it can miss crises and flag non-crises, so it must err toward showing help."""
    return bool(text) and bool(_RX.search(text))
