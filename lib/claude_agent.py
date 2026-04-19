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


# ── Admin: Generate Gap Questions ─────────────────────────────────────────────

def generate_gap_questions(
    resume_text: str,
    job_description: str,
    match_data: dict,
) -> dict:
    """
    For admin plan only. When score < 80%, generate targeted questions
    the admin can ask the candidate to surface hidden experience that
    could push the match score above 80%.
    """
    missing = match_data.get("missing", [])
    partial = match_data.get("partial", [])
    cats = match_data.get("cats", {})
    overall = match_data.get("overall", 0)
    role = match_data.get("role", "this role")

    prompt = f"""A candidate's resume scores {overall}% against a job description for {role}. The threshold is 80%.

RESUME (excerpt):
{resume_text[:2000]}

JOB DESCRIPTION (excerpt):
{job_description[:1500]}

MISSING KEYWORDS: {', '.join(missing[:12])}
PARTIAL MATCHES: {', '.join(partial[:8])}
WEAK CATEGORIES: {', '.join([k for k, v in cats.items() if v < 75])}

Generate targeted interview questions an admin can ask the candidate RIGHT NOW to surface hidden experience, projects, or skills that could close the gap to 80%+.

Rules:
- Questions must directly target the missing keywords and weak categories
- Each question should be specific enough that a "yes" answer with detail would meaningfully improve the match score
- Frame questions as conversational, not interrogative
- Include what a favorable answer would look like

Return ONLY valid JSON with no trailing commas:
{{
  "gap_summary": "One sentence explaining the main gap between the resume and JD.",
  "points_needed": {80 - overall},
  "questions": [
    {{
      "id": 1,
      "category": "Skills",
      "question": "Have you worked with [missing keyword] in any capacity, even in a personal project or supporting role?",
      "targets": ["missing_keyword"],
      "favorable_answer": "Yes, with specific examples of usage or implementation.",
      "points_if_favorable": 5
    }}
  ],
  "coaching_note": "One sentence of advice for the admin on how to use these answers."
}}

Generate 5-8 questions. Total points_if_favorable across all questions should be enough to reach 80% if answered favorably."""

    try:
        response = get_client().messages.create(
            model="claude-sonnet-4-5",
            max_tokens=2000,
            system="You are an expert resume coach and talent consultant. Return ONLY valid JSON with no trailing commas or markdown.",
            messages=[{"role": "user", "content": prompt}]
        )
        raw = response.content[0].text.strip()
        return clean_and_parse_json(raw)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate gap questions: {str(e)}")


# ── Admin: Re-analyze with Candidate Answers ──────────────────────────────────

def reanalyze_with_answers(
    resume_text: str,
    job_description: str,
    original_match: dict,
    answers: list[dict],
) -> dict:
    """
    Re-score the match incorporating the candidate's verbal answers.
    Answers can surface experience not in the resume.
    """
    answers_text = "\n".join([
        f"Q: {a.get('question', '')}\nA: {a.get('answer', '')}"
        for a in answers if a.get('answer', '').strip()
    ])

    prompt = f"""Re-analyze a candidate's match score after they answered clarifying questions.

ORIGINAL RESUME:
{resume_text[:2000]}

JOB DESCRIPTION:
{job_description[:1500]}

ORIGINAL SCORE: {original_match.get('overall')}%
ORIGINAL MISSING: {', '.join(original_match.get('missing', [])[:12])}

CANDIDATE'S ANSWERS TO GAP QUESTIONS:
{answers_text}

Instructions:
- Re-score the match incorporating the verbal answers as supplementary evidence
- If answers confirm relevant experience, increase scores accordingly
- Be honest — only increase scores for genuinely relevant answers
- The answers can be used to inform the resume summary and bullets but must not fabricate experience
- Return updated match data with the same structure as before

Return ONLY valid JSON:
{{
  "overall": <updated int 0-100>,
  "pass": <true if overall >= 80>,
  "role": "{original_match.get('role', '')}",
  "company": "{original_match.get('company', '') or ''}",
  "cats": {{
    "skills": <int>,
    "title": <int>,
    "experience": <int>,
    "keywords": <int>
  }},
  "matched": ["keyword1", "keyword2"],
  "missing": ["remaining_gap1"],
  "partial": ["partial1"],
  "tips": ["tip1", "tip2"],
  "section_analysis": [
    {{"name": "Summary", "note": "...", "status": "ok"}}
  ],
  "improvements": [
    {{
      "cat": "Keywords to add",
      "title": "Add keyword from verbal answer",
      "detail": "Candidate confirmed experience with this in their answer.",
      "where": "Skills section and relevant bullets."
    }}
  ],
  "answer_insights": "One sentence summarizing what the answers revealed.",
  "score_delta": <int, how many points were added>
}}"""

    try:
        response = get_client().messages.create(
            model="claude-sonnet-4-5",
            max_tokens=2500,
            system="You are an expert ATS analyst and resume coach. Return ONLY valid JSON with no trailing commas or markdown.",
            messages=[{"role": "user", "content": prompt}]
        )
        raw = response.content[0].text.strip()
        return clean_and_parse_json(raw)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reanalyze with answers: {str(e)}")
