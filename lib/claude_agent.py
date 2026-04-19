import anthropic
import json
import re
import os
from fastapi import HTTPException


def get_client():
    return anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


# ── JD Match Analysis ─────────────────────────────────────────────────────────

def run_jd_match(
    resume_text: str,
    job_description: str,
    transcripts: list[str] = []
) -> dict:
    """
    Analyze how well a resume matches a job description.
    Returns match score, keywords, gaps, and improvement suggestions.
    """
    client = get_client()

    transcript_context = ""
    if transcripts:
        transcript_context = "\n\nINTERVIEW TRANSCRIPTS (for context only):\n" + \
            "\n\n".join([f"Transcript {i+1}:\n{t[:1000]}" for i, t in enumerate(transcripts)])

    prompt = f"""Analyze how well this resume matches the job description. Return ONLY a raw JSON object — no markdown, no explanation.

RESUME:
{resume_text[:3000]}
{transcript_context}

JOB DESCRIPTION:
{job_description[:2000]}

Return this exact JSON structure:
{{
  "overall": <int 0-100>,
  "pass": <bool, true if overall >= 80>,
  "role": "<extracted job title>",
  "company": "<extracted company name or null>",
  "cats": {{
    "skills": <int 0-100>,
    "title": <int 0-100>,
    "experience": <int 0-100>,
    "keywords": <int 0-100>
  }},
  "matched": ["keyword1", "keyword2"],
  "missing": ["keyword1", "keyword2"],
  "partial": ["keyword1"],
  "tips": ["tip1", "tip2", "tip3"],
  "section_analysis": [
    {{"name": "Summary", "note": "...", "status": "ok"}},
    {{"name": "Skills", "note": "...", "status": "ok"}},
    {{"name": "Experience", "note": "...", "status": "warn"}},
    {{"name": "Keywords", "note": "...", "status": "ok"}}
  ],
  "improvements": [
    {{
      "cat": "Keywords to add",
      "title": "Add \\"keyword\\"",
      "detail": "...",
      "where": "Skills section and relevant bullets."
    }}
  ]
}}

Improvements should cover: missing keywords (up to 8), weak sections, formatting fixes (em dashes, bold keywords), bullet structure (4-5 per role, metric-driven), and reverse-chron ordering.
Include 8-12 total improvements."""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=2000,
            system="You are an expert ATS analyst. Return ONLY raw JSON — no markdown fences, no explanation. Start with { and end with }.",
            messages=[{"role": "user", "content": prompt}]
        )
        raw = response.content[0].text.strip()
        # Clean any markdown fences just in case
        raw = re.sub(r"```json\s*", "", raw)
        raw = re.sub(r"```\s*", "", raw)
        match = re.search(r"\{[\s\S]*\}", raw)
        if not match:
            raise ValueError("No JSON in response")
        return json.loads(match.group(0))
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse match analysis: {str(e)}")
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
    """
    Generate a fully tailored, ATS-optimized resume.
    Returns structured JSON ready for DOCX generation.
    """
    client = get_client()

    transcript_context = ""
    if transcripts:
        transcript_context = "\n\nINTERVIEW TRANSCRIPT CONTEXT (summary framing only — never fabricate bullets):\n" + \
            "\n\n".join([f"Transcript {i+1}:\n{t[:1500]}" for i, t in enumerate(transcripts)])

    improvements_list = ""
    if selected_improvements:
        improvements_list = "\nSELECTED IMPROVEMENTS TO APPLY:\n" + \
            "\n".join([f"{i+1}. {imp['title']}: {imp['detail']} ({imp['where']})"
                      for i, imp in enumerate(selected_improvements)])

    prompt = f"""Generate a tailored, ATS-optimized resume. Apply all selected improvements. Return ONLY a raw JSON object.

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

REQUIRED JSON STRUCTURE:
{{
  "name": "Full Name",
  "contact": "email · linkedin.com/in/handle · City, Country",
  "summary": "Summary with **bold** JD keywords. No em dashes.",
  "skills": ["**keyword1**", "**keyword2**", "skill3"],
  "experience": [
    {{
      "role": "Job Title",
      "company": "Company Name",
      "years": "2020–2024",
      "bullets": [
        "**Led** [what] by [X%] over [Y months] by [method].",
        "**Drove** [outcome] achieving [metric] within [timeframe].",
        "**Scaled** [what] from [A] to [B] over [period].",
        "**Reduced** [X] by [Y%] in [timeframe] through [method]."
      ]
    }}
  ],
  "education": [
    {{"degree": "Degree Name", "institution": "University", "year": "2015"}}
  ],
  "ats_score": 94
}}

Experience array: most recent first. Each role: exactly 4-5 bullets."""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=4096,
            system="You are an expert resume writer and ATS specialist. Return ONLY raw JSON — no markdown, no explanation, no fences. Strictly enforce: no em dashes, bold (**) on all JD-matched keywords, exactly 4-5 bullets per role in reverse-chronological order. Never fabricate experience.",
            messages=[{"role": "user", "content": prompt}]
        )
        raw = response.content[0].text.strip()
        raw = re.sub(r"```json\s*", "", raw)
        raw = re.sub(r"```\s*", "", raw)
        match = re.search(r"\{[\s\S]*\}", raw)
        if not match:
            raise ValueError("No JSON in response")
        return json.loads(match.group(0))
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse resume JSON: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Claude API error during generation: {str(e)}")
