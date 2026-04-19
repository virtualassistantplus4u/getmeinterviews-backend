import anthropic as _sdk
import json
import re
import os
from fastapi import HTTPException

_MODEL = "claude-sonnet-4-5"

def get_client():
    return _sdk.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

# ── Admin system prompt ───────────────────────────────────────────────────────
ENGINE_SYSTEM_PROMPT = """You are Resume Agent, an expert ATS analyst and resume writer.
Return ONLY raw valid JSON — no markdown fences, no explanation, no trailing commas.
Start with { and end with }.
Scoring weights: Skills match 40%, Title alignment 25%, Experience depth 20%, Keyword density 15%.
Be precise and consistent — your scores must reflect genuine keyword and experience overlap."""

# ── JSON parser ───────────────────────────────────────────────────────────────
def clean_and_parse_json(raw: str) -> dict:
    raw = re.sub(r"```json\s*", "", raw)
    raw = re.sub(r"```\s*", "", raw)
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    start = raw.find("{")
    if start == -1:
        raise HTTPException(status_code=500, detail="No valid response object found.")
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
        raise HTTPException(status_code=500, detail="Could not find complete response object.")
    candidate = raw[start:end]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
    candidate = re.sub(r'(?<!\\)\n', ' ', candidate)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse AI response: {str(e)}")

# ── Local scoring (Free plan) ─────────────────────────────────────────────────
def run_jd_match_local(resume_text: str, job_description: str, transcripts: list[str] = []) -> dict:
    """Fast local keyword scoring for free plan users."""
    jl = job_description.lower()
    rt = (resume_text or "").lower()
    tt = " ".join(t.lower() for t in transcripts)

    stop = {"their","about","which","there","these","those","where","after","before","other",
            "would","could","should","being","having","doing","using","making","taking","giving",
            "going","coming","getting","putting","saying","seeing","finding","working","looking",
            "keeping","trying","calling","asking","needing","able","will","can","all","its",
            "into","more","also","any","when","than","your","our","the","and","for","are",
            "was","been","with","that","this","from","they","have","had","not","but","role",
            "team","work","strong","great","good","best","well","provide","ensure","across",
            "within","including","requirements","experience","skills","ability","company","position"}

    def kwds(txt):
        w = re.split(r'\W+', txt.lower())
        f = {}
        for x in w:
            if len(x) > 4 and x not in stop:
                f[x] = f.get(x, 0) + 1
        return [k for k, _ in sorted(f.items(), key=lambda x: -x[1])[:80]]

    jKws = kwds(job_description)
    matched, missing, partial = [], [], []
    smap = {}
    for k in jKws:
        inR = k in rt
        inT = k in tt
        if inR or inT:
            matched.append(k)
            smap[k] = "both" if (inR and inT) else ("resume" if inR else "transcript")
        else:
            ph = any(p.startswith(k[:5]) or k.startswith(p[:5]) for p in kwds(rt + " " + tt))
            if ph:
                partial.append(k)
                smap[k] = "partial"
            else:
                missing.append(k)

    skill_score = min(100, round((len(matched) / max(len(jKws), 1)) * 135))
    title_kws = ["senior","lead","principal","staff","director","manager","engineer",
                 "analyst","designer","developer","scientist","head"]
    title_ok = any(k in jl and (k in rt or k in tt) for k in title_kws)
    exp_ok = len(resume_text) > 500
    t_bonus = min(10, len(transcripts) * 3)
    cats = {
        "skills": min(100, skill_score),
        "title": 84 if title_ok else 50,
        "experience": 87 if exp_ok else 60,
        "keywords": min(100, round((len(matched) + len(partial) * 0.5) / max(len(jKws), 1) * 120))
    }
    overall = min(97, round(cats["skills"] * 0.40 + cats["title"] * 0.25 + cats["experience"] * 0.20 + cats["keywords"] * 0.15 + t_bonus))

    rm = re.search(r'(Senior|Staff|Lead|Principal|Director|Head of|VP[\w\s]+|[\w ]{3,28}(?:Manager|Engineer|Designer|Analyst|Scientist|Developer|PM))', job_description, re.I)
    cm = re.search(r'(?:at|join|joining|for)\s+([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)?)', job_description)

    improvements = []
    for kw in missing[:8]:
        improvements.append({"cat": "Keywords to add", "title": f'Add "{kw}"', "detail": f'JD keyword missing from profile.', "where": "Skills section and most relevant bullet."})
    if cats["skills"] < 80:
        improvements.append({"cat": "Sections to strengthen", "title": "Expand skills section", "detail": f'Skills coverage is {cats["skills"]}%.', "where": "Move matched JD keywords to top of Skills."})
    if cats["title"] < 75:
        improvements.append({"cat": "Sections to strengthen", "title": "Align title to JD seniority", "detail": "Title does not closely match JD.", "where": "First sentence of Summary."})
    improvements += [
        {"cat": "Sections to strengthen", "title": "Rewrite summary", "detail": "Open with JD title + top 3 keywords.", "where": "Professional Summary."},
        {"cat": "Formatting fixes", "title": "Remove all em dashes", "detail": "Em dashes confuse ATS parsers.", "where": "Every bullet and sentence."},
        {"cat": "Formatting fixes", "title": "Bold JD-matched keywords", "detail": "All matched terms marked bold for ATS.", "where": "Skills and bullets."},
        {"cat": "Content and structure", "title": "Exactly 4-5 bullets per role", "detail": "Each role: 4-5 focused bullets.", "where": "All Experience roles."},
        {"cat": "Content and structure", "title": "Metric-driven bullet structure", "detail": "Action verb + % metric + time + method.", "where": "All experience bullets."},
    ]

    return {
        "overall": overall, "pass": overall >= 80,
        "role": rm.group(1).strip() if rm else "this role",
        "company": cm.group(1) if cm else None,
        "cats": cats,
        "matched": matched[:16], "missing": missing[:12], "partial": partial[:8],
        "tips": [f'Mirror: {", ".join(missing[:4])}', "Every bullet: action verb + metric + time.", "Bold all JD keywords."],
        "section_analysis": [
            {"name": "Summary", "note": "Rewrite to open with JD title.", "status": "warn" if overall < 80 else "ok"},
            {"name": "Skills", "note": f'{len(matched)} matches found.', "status": "ok" if len(matched) > 6 else "warn"},
            {"name": "Experience", "note": "4-5 bullets per role, reverse-chron.", "status": "ok"},
            {"name": "ATS Format", "note": "Calibri, single-column, no em dashes.", "status": "ok"},
        ],
        "improvements": improvements
    }

def _run_match_deep(resume_text: str, job_description: str, transcripts: list[str] = []) -> dict:
    """Deep scoring engine for paid plans."""
    transcript_context = ""
    if transcripts:
        transcript_context = "\n\nINTERVIEW TRANSCRIPTS:\n" + "\n\n".join([f"Transcript {i+1}:\n{t[:1000]}" for i, t in enumerate(transcripts)])

    prompt = f"""Analyze how well this resume matches the job description using these exact weights:
- Skills match: 40% (keyword overlap between profile skills and JD requirements)
- Title alignment: 25% (seniority and role type match)
- Experience depth: 20% (years, industry, scope vs JD requirements)
- Keyword density: 15% (JD-specific terminology coverage)

RESUME:
{resume_text[:3000]}
{transcript_context}

JOB DESCRIPTION:
{job_description[:2000]}

Score carefully and honestly. Be consistent — the same resume and JD should always produce the same score.

Return ONLY valid JSON with no trailing commas:
{{
  "overall": 75,
  "pass": false,
  "role": "extracted job title from JD",
  "company": "extracted company name or null",
  "cats": {{
    "skills": 70,
    "title": 80,
    "experience": 75,
    "keywords": 65
  }},
  "matched": ["keyword1", "keyword2"],
  "missing": ["gap1", "gap2"],
  "partial": ["partial1"],
  "tips": ["Specific tip 1", "Specific tip 2", "Specific tip 3"],
  "section_analysis": [
    {{"name": "Summary", "note": "Specific actionable note.", "status": "warn"}},
    {{"name": "Skills", "note": "Specific note.", "status": "ok"}},
    {{"name": "Experience", "note": "Specific note.", "status": "ok"}},
    {{"name": "ATS Format", "note": "Calibri, single-column, no em dashes.", "status": "ok"}}
  ],
  "improvements": [
    {{
      "cat": "Keywords to add",
      "title": "Add \\"specific keyword\\"",
      "detail": "This JD keyword is missing. Add where truthful.",
      "where": "Skills section and most relevant experience bullet."
    }}
  ]
}}

Rules:
- overall = round(skills*0.40 + title*0.25 + experience*0.20 + keywords*0.15)
- pass = true only if overall >= 80
- matched: up to 16 keywords genuinely found in resume
- missing: up to 12 keywords genuinely absent
- improvements: 8-12 items covering keywords, section rewrites, formatting, bullet structure
- NO trailing commas"""

    try:
        response = get_client().messages.create(
            model=_MODEL,
            max_tokens=2500,
            system=ENGINE_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}]
        )
        return clean_and_parse_json(response.content[0].text.strip())
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI scoring error: {str(e)}")

def run_jd_match(resume_text: str, job_description: str, transcripts: list[str] = [], plan: str = "free") -> dict:
    """Route to correct scoring engine based on plan."""
    if plan in ("pro", "unlimited", "admin"):
        return _run_match_deep(resume_text, job_description, transcripts)
    return run_jd_match_local(resume_text, job_description, transcripts)

# ── Resume Generation ─────────────────────────────────────────────────────────
def generate_resume(resume_text: str, job_description: str, match_data: dict, selected_improvements: list[dict], transcripts: list[str] = []) -> dict:
    transcript_context = ""
    if transcripts:
        transcript_context = "\n\nINTERVIEW TRANSCRIPT CONTEXT (summary framing only):\n" + "\n\n".join([f"Transcript {i+1}:\n{t[:1500]}" for i, t in enumerate(transcripts)])

    improvements_list = ""
    if selected_improvements:
        improvements_list = "\nSELECTED IMPROVEMENTS:\n" + "\n".join([f"{i+1}. {imp['title']}: {imp['detail']} ({imp['where']})" for i, imp in enumerate(selected_improvements)])

    prompt = f"""Generate a tailored ATS-optimized resume. Return ONLY valid JSON with no trailing commas.

MASTER RESUME:
{resume_text[:3000]}
{transcript_context}

JOB DESCRIPTION:
{job_description[:2000]}

MATCH DATA:
Score: {match_data.get('overall')}% | Role: {match_data.get('role')}{' at ' + match_data.get('company') if match_data.get('company') else ''}
Matched: {', '.join(match_data.get('matched', []))}
Missing: {', '.join(match_data.get('missing', []))}
{improvements_list}

ABSOLUTE RULES:
1. Single-column only. No tables, columns, images.
2. NEVER use em dash (—). Use comma or colon instead.
3. Mark ALL JD-matched keywords with **double asterisks** for bold.
4. Experience: MOST RECENT role FIRST (reverse-chronological).
5. Every role: EXACTLY 4 to 5 bullet points.
6. Every bullet: action verb + metric + time period + method.
7. Mirror exact JD phrases. Nothing fabricated.
8. Summary opens with exact JD job title + top 3 matched keywords.

Return this exact JSON:
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
  "education": [{{"degree": "Degree", "institution": "University", "year": "2015"}}],
  "ats_score": 94
}}"""

    try:
        response = get_client().messages.create(
            model=_MODEL,
            max_tokens=4096,
            system="You are an expert resume writer. Return ONLY valid JSON. No markdown, no em dashes, exactly 4-5 bullets per role in reverse-chronological order. Never fabricate.",
            messages=[{"role": "user", "content": prompt}]
        )
        return clean_and_parse_json(response.content[0].text.strip())
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI generation error: {str(e)}")

# ── Admin: Gap Questions (multi-round) ────────────────────────────────────────
def generate_gap_questions(resume_text: str, job_description: str, match_data: dict,
                           round_number: int = 1, previous_answers: list[dict] = []) -> dict:
    """
    Generate targeted gap questions for admin plan.
    Supports multiple rounds — returns can_generate_more flag.
    """
    missing = match_data.get("missing", [])
    partial = match_data.get("partial", [])
    cats = match_data.get("cats", {})
    overall = match_data.get("overall", 0)
    role = match_data.get("role", "this role")
    points_needed = 80 - overall

    prev_context = ""
    if previous_answers:
        prev_context = "\n\nPREVIOUS ROUND ANSWERS (already asked — do NOT repeat these questions):\n" + \
            "\n".join([f"Q: {a.get('question', '')}\nA: {a.get('answer', '')}" for a in previous_answers])

    prompt = f"""A candidate scores {overall}% for {role}. Need {points_needed} more points to reach 80%.
Round {round_number} of 2 maximum.

RESUME (excerpt):
{resume_text[:1500]}

JOB DESCRIPTION (excerpt):
{job_description[:1000]}

REMAINING GAPS: {', '.join(missing[:10])}
PARTIAL MATCHES: {', '.join(partial[:6])}
WEAK CATEGORIES: {', '.join([k for k, v in cats.items() if v < 75])}
{prev_context}

Your task: Generate targeted questions to surface hidden experience that could close the gap.

IMPORTANT: First assess whether meaningful new questions can be generated:
- If the gaps are highly specialized (e.g. "German language", specific certifications, very niche tools) that cannot be bridged by asking questions, set can_generate_more to false
- If round_number is already 2, set can_generate_more to false regardless
- If there are still genuine experience gaps that questions could uncover, set can_generate_more to true

Return ONLY valid JSON:
{{
  "gap_summary": "One sentence explaining the main remaining gap.",
  "points_needed": {points_needed},
  "round": {round_number},
  "can_generate_more": false,
  "no_more_reason": "Explain why no further questions can help (only if can_generate_more is false)",
  "questions": [
    {{
      "id": 1,
      "category": "Skills",
      "question": "Conversational question targeting the gap...",
      "targets": ["missing_keyword"],
      "favorable_answer": "What a good answer looks like.",
      "points_if_favorable": 5
    }}
  ],
  "coaching_note": "One sentence advice for the admin on how to use these answers."
}}

Generate 4-7 questions. Do NOT repeat any questions from previous rounds.
Set can_generate_more based on whether a hypothetical round {round_number + 1} could still help."""

    try:
        response = get_client().messages.create(
            model=_MODEL,
            max_tokens=2000,
            system="You are an expert resume coach. Return ONLY valid JSON with no trailing commas or markdown.",
            messages=[{"role": "user", "content": prompt}]
        )
        return clean_and_parse_json(response.content[0].text.strip())
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gap questions error: {str(e)}")

# ── Admin: Re-analyze with Answers ────────────────────────────────────────────
def reanalyze_with_answers(resume_text: str, job_description: str, original_match: dict, answers: list[dict]) -> dict:
    answers_text = "\n".join([f"Q: {a.get('question', '')}\nA: {a.get('answer', '')}" for a in answers if a.get('answer', '').strip()])

    prompt = f"""Re-score a resume match after the candidate answered clarifying questions.

ORIGINAL RESUME:
{resume_text[:2000]}

JOB DESCRIPTION:
{job_description[:1500]}

ORIGINAL SCORE: {original_match.get('overall')}%
ORIGINAL MISSING: {', '.join(original_match.get('missing', [])[:12])}

CANDIDATE ANSWERS:
{answers_text}

Instructions:
- Re-score using the same weights: Skills 40%, Title 25%, Experience 20%, Keywords 15%
- Increase scores only where answers genuinely confirm relevant experience
- Be honest — do not inflate scores beyond what the answers justify
- Update matched/missing based on what the answers revealed

Return ONLY valid JSON:
{{
  "overall": 75,
  "pass": false,
  "role": "{original_match.get('role', '')}",
  "company": "{original_match.get('company', '') or ''}",
  "cats": {{"skills": 80, "title": 70, "experience": 75, "keywords": 65}},
  "matched": ["keyword1"],
  "missing": ["remaining_gap1"],
  "partial": ["partial1"],
  "tips": ["tip1", "tip2"],
  "section_analysis": [{{"name": "Summary", "note": "...", "status": "ok"}}],
  "improvements": [
    {{
      "cat": "Keywords to add",
      "title": "Add keyword confirmed in answers",
      "detail": "Candidate confirmed this in their verbal answer.",
      "where": "Skills section and relevant bullets."
    }}
  ],
  "answer_insights": "One sentence summarizing what the answers revealed.",
  "score_delta": 8
}}"""

    try:
        response = get_client().messages.create(
            model=_MODEL,
            max_tokens=2500,
            system=ENGINE_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}]
        )
        return clean_and_parse_json(response.content[0].text.strip())
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reanalysis error: {str(e)}")
