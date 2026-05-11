import os
import re
import json
import anthropic

def _get_client():
    return anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))

# ── Rule-based patterns ──────────────────────────────────────────────────────

_INVESTOR = re.compile(
    r'\b(vc|venture capital|venture partner|general partner|managing partner'
    r'|investment|investor|angel investor|angel|fund manager|portfolio manager'
    r'|private equity|principal.*fund|partner.*capital|capital.*partner'
    r'|limited partner|family office|endowment|allocator|lp\b)\b',
    re.I
)
_FOUNDER = re.compile(
    r'\b(founder|co-founder|cofounder|ceo|chief executive|cto|chief technology'
    r'|coo|chief operating|cpo|chief product|cmo|chief marketing'
    r'|president|owner|proprietor|entrepreneur)\b',
    re.I
)
_CREATOR = re.compile(
    r'\b(creator|content creator|influencer|youtuber|tiktoker|podcaster'
    r'|streamer|blogger|author|newsletter|journalist|media|correspondent'
    r'|host\b|anchor)\b',
    re.I
)
_OPERATOR = re.compile(
    r'\b(vp|vice president|director|head of|manager|lead\b|principal engineer'
    r'|senior engineer|staff engineer|product manager|pm\b|engineering manager'
    r'|sales|marketing|operations|growth)\b',
    re.I
)

# Industry / sector tags
_TAG_PATTERNS = {
    "fintech": r'\b(fintech|finance|banking|payments|crypto|defi|web3|blockchain)\b',
    "saas":    r'\b(saas|software|b2b software|enterprise software|cloud)\b',
    "ai/ml":   r'\b(ai|ml|machine learning|deep learning|llm|nlp|artificial intelligence)\b',
    "health":  r'\b(health|medtech|biotech|healthcare|pharma|clinical)\b',
    "consumer":r'\b(consumer|d2c|dtc|ecommerce|retail|marketplace)\b',
    "climate": r'\b(climate|cleantech|green|sustainability|energy|solar)\b',
    "media":   r'\b(media|entertainment|gaming|sports|music|film)\b',
    "deep tech":r'\b(deep.?tech|hardware|robotics|aerospace|defense|biotech)\b',
    "real estate": r'\b(real estate|proptech|realty|property)\b',
    "crypto":  r'\b(crypto|web3|nft|defi|blockchain|dao)\b',
}


def rule_based_tag(member: dict) -> tuple[str, float, list[str]]:
    """
    Returns (member_type, confidence, tags).
    confidence 0-1: how sure we are about the type.
    """
    text = " ".join(filter(None, [
        member.get("title", ""),
        member.get("company", ""),
        member.get("bio", ""),
    ])).lower()

    scores = {
        "investor": len(_INVESTOR.findall(text)),
        "founder":  len(_FOUNDER.findall(text)),
        "creator":  len(_CREATOR.findall(text)),
        "operator": len(_OPERATOR.findall(text)),
    }

    # Detect tags
    tags = []
    for tag, pattern in _TAG_PATTERNS.items():
        if re.search(pattern, text, re.I):
            tags.append(tag)

    best_type = max(scores, key=scores.get)
    best_score = scores[best_type]

    if best_score == 0:
        return "unknown", 0.0, tags

    total = sum(scores.values())
    confidence = best_score / total if total > 0 else 0.0

    # Investor + founder both match → likely investor-founder (keep investor)
    if scores["investor"] > 0 and scores["founder"] > 0:
        best_type = "investor" if scores["investor"] >= scores["founder"] else "founder"

    # Operator patterns overlap with founder → if both match, trust founder
    if best_type == "operator" and scores["founder"] > 0:
        best_type = "founder"
        confidence = scores["founder"] / total

    return best_type, min(confidence * 1.2, 1.0), tags


def batch_tag_with_claude(members: list[dict]) -> list[dict]:
    """
    Send a batch of low-confidence members to Claude for classification.
    Returns members with updated member_type, confidence, tags.
    """
    if not members:
        return members

    payload = [
        {
            "id": m["_batch_id"],
            "name": m.get("name", ""),
            "title": m.get("title", ""),
            "company": m.get("company", ""),
            "bio": (m.get("bio") or "")[:300],
        }
        for m in members
    ]

    prompt = f"""Classify each person by member_type and extract industry tags.

member_type options:
- founder: Started or owns a company (CEO/CTO/CXO of their own company, co-founder)
- investor: Invests money in companies (VC partner, angel investor, fund manager, family office)
- creator: Creates content for an audience (influencer, podcaster, blogger, newsletter writer, YouTuber)
- operator: Works at a company but didn't found it (VP, Director, manager, individual contributor)
- other: Speaker, consultant, advisor, academic, government

Tags: list up to 5 relevant tags from: fintech, saas, ai/ml, health, consumer, climate, media, deep tech, real estate, crypto, b2b, b2c, early-stage, growth-stage, enterprise

Members:
{json.dumps(payload, indent=2)}

Return ONLY a JSON array (no markdown):
[{{"id": 0, "member_type": "founder", "confidence": 0.9, "tags": ["saas", "b2b"]}}]"""

    try:
        response = _get_client().messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip()
        if "```" in text:
            m = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
            if m:
                text = m.group(1)
        results = json.loads(text)

        result_map = {r["id"]: r for r in results}
        for member in members:
            r = result_map.get(member["_batch_id"])
            if r:
                member["member_type"] = r.get("member_type", member.get("member_type", "unknown"))
                member["confidence"] = float(r.get("confidence", 0.7))
                claude_tags = r.get("tags", [])
                member["tags"] = list(set(member.get("tags", []) + claude_tags))
    except Exception as e:
        print(f"[tagger] Claude batch failed: {e}")

    return members


def tag_members(members: list[dict]) -> list[dict]:
    """Tag all members: rule-based first, Claude for uncertain ones."""
    for member in members:
        mtype, conf, tags = rule_based_tag(member)
        member["member_type"] = mtype
        member["confidence"] = conf
        member["tags"] = list(set(member.get("tags", []) + tags))

    # Send uncertain members to Claude in batches of 40
    uncertain = [m for m in members if m["confidence"] < 0.6]
    for i in range(0, len(uncertain), 40):
        batch = uncertain[i:i+40]
        for j, m in enumerate(batch):
            m["_batch_id"] = j
        batch_tag_with_claude(batch)

    # Clean up temp field
    for m in members:
        m.pop("_batch_id", None)

    return members
