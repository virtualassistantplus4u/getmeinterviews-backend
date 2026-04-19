import anthropic
import json
import re
import os
from fastapi import HTTPException

# ── Admin system prompt (internal plan only) ─────────────────────────────────
ADMIN_SYSTEM_PROMPT = """You are Resume Agent, an AI assistant that tailors resumes to job descriptions, scores keyword matches, and generates ATS-optimized DOCX output. You follow a strict, consistent workflow.

ABSOLUTE RULES:
1. Single-column only. No tables, columns, images, text boxes.
2. NEVER use an em dash (—) anywhere. Use a comma or colon instead.
3. Mark ALL JD-matched keywords with **double asterisks** for bold.
4. Experience: MOST RECENT role FIRST (reverse-chronological).
5. Every role: EXACTLY 4 to 5 bullet points — no more, no fewer.
6. Every bullet: unique past-tense action verb + [what improved] by [X%] over [Y period] by [method].
7. Mirror exact JD phrases — do not paraphrase.
8. Summary opens with the exact JD job title + top 3 matched keywords.
9. Skills section: matched JD keywords listed first.
10. Nothing fabricated. Every bullet grounded in the master resume.
11. Return ONLY raw JSON — no markdown fences, no explanation."""



def get_client():
    return anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


def clean_and_parse_json(raw: str) -> dict:
    """Robustly parse JSON from Claude's response."""
    # Remove markdown fences
    raw = re.sub(r"```json\s*", "", raw)
    raw = re.sub(r"```\s*", "", raw)
    raw = raw.strip()

    # Try direct parse first
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Find the outermost { ... } block
    start = raw.find("{")
    if start == -1:
        raise HTTPException(status_code=500, detail="No JSON object found in Claude response.")

    # Walk through to find matching closing brace
    depth = 0
    end = -1
    in_string = False
    escape_next = False
    for i, ch in enumerate(raw[start:], start):
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break

    if end == -1:
        raise HTTPException(status_code=500, detail="Could not find complete JSON in Claude response.")

    candidate = raw[start:end]

    # Try parsing the extracted block
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # Last resort: fix common Claude JSON issues
    # Remove trailing commas before } or ]
    candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
    # Fix unescaped newlines inside strings
    candidate = re.sub(r'(?<!\\)\n', ' ', candidate)

    try:
        return json.loads(candidate)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse Claude response as JSON: {str(e)}")


# ── JD Match Analysis ─────────────────────────────────────────────────────────

def run_jd_match(
    resume_text: str,
    job_description: str,
    transcripts: list[str] = []
) -> dict:
    client = get_client()

    transcript_context = ""
    if transcripts:
        transcript_context = "\n\nINTERVIEW TRANSCRIPTS (for context only):\n" + \
            "\n\n".join([f"Transcript {i+1}:\n{t[:1000]}" for i, t in enumerate(transcripts)])

    prompt = f"""Analyze how well this resume matches the job description. Return ONLY a raw JSON object — no markdown, no explanation, no trailing commas.

RESUME:
{resume_text[:3000]}
{transcript_context}

JOB DESCRIPTION:
{job_description[:2000]}

Return this exact JSON structure with no trailing commas:
{{
  "overall": 75,
  "pass": true,
  "role": "Software Engineer",
  "company": "Acme Corp",
  "cats": {{
    "skills": 80,
    "title": 70,
    "experience": 75,
    "keywords": 72
  }},
  "matched": ["python", "fastapi"],
  "missing": ["kubernetes", "terraform"],
  "partial": ["docker"],
  "tips": ["Mirror JD phrases exactly", "Add missing keywords where truthful"],
  "section_analysis": [
    {{"name": "Summary", "note": "Rewrite to open with JD title", "status": "warn"}},
    {{"name": "Skills", "note": "Move matched keywords first", "status": "ok"}}
  ],
  "improvements": [
    {{
      "cat": "Keywords to add",
      "title": "Add \\"kubernetes\\"",
      "detail": "Missing JD keyword. Add where truthful.",
      "where": "Skills section and relevant bullets."
    }}
  ]
}}

Rules:
- overall is 0-100 integer
- pass is true if overall >= 80
- matched: up to 16 keywords found in resume
- missing: up to 12 keywords not found
- improvements: 8-12 items covering keywords, section rewrites, formatting fixes, bullet structure
- NO trailing commas anywhere in the JSON"""

    try:
        response = get_client().messages.create(
            model="claude-sonnet-4-5",
            max_tokens=2500,
            system="You are an expert ATS analyst. Return ONLY valid JSON with no trailing commas, no markdown fences, no explanation. Start with { and end with }.",
            messages=[{"role": "user", "content": prompt}]
        )
        raw = response.content[0].text.strip()
        return clean_and_parse_json(raw)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Claude API error during match: {str(e)}")


# ── Resume Generation ─────────────────────────────────────────────────────────

def generate_resume(
    resume_text: str,
    job_description: str,
    match_data: dict,
    selected_improvements: list[dict],
    transcripts: list[str] = []
) -> dict:
    client = get_client()

    transcript_context = ""
    if transcripts:
        transcript_context = "\n\nINTERVIEW TRANSCRIPT CONTEXT (summary framing only):\n" + \
            "\n\n".join([f"Transcript {i+1}:\n{t[:1500]}" for i, t in enumerate(transcripts)])

    improvements_list = ""
    if selected_improvements:
        improvements_list = "\nSELECTED IMPROVEMENTS TO APPLY:\n" + \
            "\n".join([f"{i+1}. {imp['title']}: {imp['detail']} ({imp['where']})"
                      for i, imp in enumerate(selected_improvements)])

    prompt = f"""Generate a tailored ATS-optimized resume. Return ONLY valid JSON with no trailing commas.

MASTER RESUME:
{resume_text[:3000]}
{transcript_context}

JOB DESCRIPTION:
{job_description[:2000]}

MATCH DATA:
Score: {match_data.get('overall')}% | Role: {match_data.get('role')}{' at ' + match_data.get('company') if match_data.get('company') else ''}
Matched keywords: {', '.join(match_data.get('matched', []))}
Missing keywords: {', '.join(match_data.get('missing', []))}
{improvements_list}

ABSOLUTE RULES:
1. Single-column only. No tables, columns, images.
2. NEVER use em dash (—). Use comma or colon instead.
3. Mark ALL JD-matched keywords with **double asterisks** for bold.
4. Experience: MOST RECENT role FIRST (reverse-chronological).
5. Every role: EXACTLY 4 to 5 bullet points.
6. Every bullet: action verb + metric + time period + method.
7. Mirror exact JD phrases.
8. Summary opens with exact JD job title + top 3 matched keywords.
9. Nothing fabricated.

Return this exact JSON with no trailing commas:
{{
  "name": "Full Name",
  "contact": "email · linkedin · City, Country",
  "summary": "Summary with **bold** keywords. No em dashes.",
  "skills": ["**keyword1**", "**keyword2**", "skill3"],
  "experience": [
    {{
      "role": "Job Title",
      "company": "Company",
      "years": "2020-2024",
      "bullets": [
        "**Led** X by Y% over Z months by method.",
        "**Drove** outcome achieving metric.",
        "**Scaled** from A to B over period.",
        "**Reduced** X by Y% through method."
      ]
    }}
  ],
  "education": [
    {{"degree": "Degree", "institution": "University", "year": "2015"}}
  ],
  "ats_score": 94
}}"""

    try:
        response = get_client().messages.create(
            model="claude-sonnet-4-5",
            max_tokens=4096,
            system="You are an expert resume writer. Return ONLY valid JSON with no trailing commas, no markdown fences. Never fabricate experience. No em dashes anywhere.",
            messages=[{"role": "user", "content": prompt}]
        )
        raw = response.content[0].text.strip()
        return clean_and_parse_json(raw)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Claude API error during generation: {str(e)}")
