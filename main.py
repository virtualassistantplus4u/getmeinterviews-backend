from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
import os
import re
from dotenv import load_dotenv

load_dotenv()

from lib.auth import get_current_user
from lib.text_extractor import extract_text
from lib.claude_agent import run_jd_match, generate_resume
from lib.docx_generator import generate_docx

app = FastAPI(title="GetMeInterviews API", version="1.0.0")

# ── CORS ──────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        os.getenv("FRONTEND_URL", "http://localhost:3000"),
        "https://get-me-interviews.vercel.app",
        "https://getmeinterviews.com",
        "https://www.getmeinterviews.com",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Plan limits ───────────────────────────────────────────────
PLAN_LIMITS = {
    "free":      {"monthly_resumes": 3,       "transcripts": 0, "daily_jd_matches": 3},
    "pro":       {"monthly_resumes": 20,      "transcripts": 1, "daily_jd_matches": 10},
    "unlimited": {"monthly_resumes": 999999,  "transcripts": 5, "daily_jd_matches": 999999},
    "admin":     {"monthly_resumes": 999999,  "transcripts": 999999, "daily_jd_matches": 999999},
}


def check_jd_limit(profile: dict):
    plan = profile.get("plan", "free")
    limit = PLAN_LIMITS[plan]["daily_jd_matches"]
    used = profile.get("jd_matches_today", 0)
    if used >= limit:
        raise HTTPException(
            status_code=429,
            detail=f"Daily JD match limit reached ({limit}/day on {plan.title()} plan). Resets at midnight."
        )


def check_resume_limit(profile: dict):
    plan = profile.get("plan", "free")
    limit = PLAN_LIMITS[plan]["monthly_resumes"]
    used = profile.get("resumes_used_this_month", 0)
    if used >= limit:
        raise HTTPException(
            status_code=429,
            detail=f"Monthly resume generation limit reached ({limit}/month on {plan.title()} plan)."
        )


def check_transcript_limit(profile: dict, current_count: int):
    plan = profile.get("plan", "free")
    limit = PLAN_LIMITS[plan]["transcripts"]
    if limit == 0:
        raise HTTPException(status_code=403, detail="Interview transcripts are not available on the Free plan. Upgrade to Pro.")
    if current_count >= limit:
        raise HTTPException(status_code=403, detail=f"Transcript limit reached ({limit} on {plan.title()} plan). Delete one to upload a new one.")


# ── Health check ──────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "service": "GetMeInterviews API"}


# ── Upload master resume ──────────────────────────────────────
@app.post("/api/resume/upload")
async def upload_resume(
    file: UploadFile = File(...),
    ctx: dict = Depends(get_current_user)
):
    profile = ctx["profile"]
    supabase = ctx["supabase"]
    user_id = profile["id"]

    # Check if user already has a master resume
    existing = supabase.table("resumes").select("id").eq("user_id", user_id).eq("is_master", True).execute()
    if existing.data:
        raise HTTPException(
            status_code=400,
            detail="You already have a master resume uploaded. Delete it first to upload a new one."
        )

    # Validate file type
    if not file.filename.lower().endswith((".pdf", ".docx")):
        raise HTTPException(status_code=422, detail="Only PDF and DOCX files are supported.")

    # Validate file size (5MB max)
    content = await file.read()
    if len(content) > 5 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large. Maximum size is 5MB.")

    # Extract text
    raw_text = extract_text(content, file.filename)

    # Save to Supabase
    result = supabase.table("resumes").insert({
        "user_id": user_id,
        "file_name": file.filename,
        "file_size": len(content),
        "raw_text": raw_text,
        "is_master": True,
    }).execute()

    return {
        "id": result.data[0]["id"],
        "file_name": file.filename,
        "file_size": len(content),
        "text_length": len(raw_text),
        "message": "Resume uploaded and text extracted successfully."
    }


# ── Delete master resume ──────────────────────────────────────
@app.delete("/api/resume/{resume_id}")
async def delete_resume(
    resume_id: str,
    ctx: dict = Depends(get_current_user)
):
    supabase = ctx["supabase"]
    user_id = ctx["profile"]["id"]

    result = supabase.table("resumes")\
        .delete()\
        .eq("id", resume_id)\
        .eq("user_id", user_id)\
        .execute()

    if not result.data:
        raise HTTPException(status_code=404, detail="Resume not found.")

    return {"message": "Resume deleted successfully."}


# ── Upload transcript ─────────────────────────────────────────
@app.post("/api/transcript/upload")
async def upload_transcript(
    file: UploadFile = File(...),
    ctx: dict = Depends(get_current_user)
):
    profile = ctx["profile"]
    supabase = ctx["supabase"]
    user_id = profile["id"]

    # Check transcript limit
    existing = supabase.table("transcripts").select("id").eq("user_id", user_id).execute()
    check_transcript_limit(profile, len(existing.data or []))

    # Validate file
    if not file.filename.lower().endswith((".pdf", ".docx", ".txt")):
        raise HTTPException(status_code=422, detail="Only PDF, DOCX, and TXT files are supported for transcripts.")

    content = await file.read()
    if len(content) > 5 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large. Maximum size is 5MB.")

    # Extract text
    if file.filename.lower().endswith(".txt"):
        raw_text = content.decode("utf-8", errors="ignore")
    else:
        raw_text = extract_text(content, file.filename)

    result = supabase.table("transcripts").insert({
        "user_id": user_id,
        "file_name": file.filename,
        "raw_text": raw_text,
    }).execute()

    return {
        "id": result.data[0]["id"],
        "file_name": file.filename,
        "text_length": len(raw_text),
        "message": "Transcript uploaded successfully."
    }


# ── Delete transcript ─────────────────────────────────────────
@app.delete("/api/transcript/{transcript_id}")
async def delete_transcript(
    transcript_id: str,
    ctx: dict = Depends(get_current_user)
):
    supabase = ctx["supabase"]
    user_id = ctx["profile"]["id"]

    result = supabase.table("transcripts")\
        .delete()\
        .eq("id", transcript_id)\
        .eq("user_id", user_id)\
        .execute()

    if not result.data:
        raise HTTPException(status_code=404, detail="Transcript not found.")

    return {"message": "Transcript deleted successfully."}


# ── JD Match ──────────────────────────────────────────────────
class MatchRequest(BaseModel):
    job_description: str
    resume_id: str | None = None


@app.post("/api/match")
async def match_jd(
    body: MatchRequest,
    ctx: dict = Depends(get_current_user)
):
    profile = ctx["profile"]
    supabase = ctx["supabase"]
    user_id = profile["id"]

    # Check daily JD match limit
    check_jd_limit(profile)

    if not body.job_description or len(body.job_description.strip()) < 100:
        raise HTTPException(status_code=422, detail="Job description is too short. Please paste the full JD.")

    # Get master resume
    resume_res = supabase.table("resumes")\
        .select("*")\
        .eq("user_id", user_id)\
        .eq("is_master", True)\
        .single()\
        .execute()

    if not resume_res.data:
        raise HTTPException(status_code=404, detail="No master resume found. Please upload your resume first.")

    # Get transcripts
    transcripts_res = supabase.table("transcripts")\
        .select("raw_text")\
        .eq("user_id", user_id)\
        .execute()

    transcript_texts = [t["raw_text"] for t in (transcripts_res.data or []) if t.get("raw_text")]

    # Run match analysis
    match_result = run_jd_match(
        resume_text=resume_res.data["raw_text"],
        job_description=body.job_description,
        transcripts=transcript_texts
    )

    # Increment daily JD match counter
    supabase.rpc("increment_jd_matches", {"user_id": user_id}).execute()

    return match_result


# ── Generate Resume ───────────────────────────────────────────
class GenerateRequest(BaseModel):
    job_description: str
    match_data: dict
    selected_improvements: list[dict]


@app.post("/api/generate")
async def generate(
    body: GenerateRequest,
    ctx: dict = Depends(get_current_user)
):
    profile = ctx["profile"]
    supabase = ctx["supabase"]
    user_id = profile["id"]

    # Check monthly generation limit
    check_resume_limit(profile)

    # Get master resume
    resume_res = supabase.table("resumes")\
        .select("*")\
        .eq("user_id", user_id)\
        .eq("is_master", True)\
        .single()\
        .execute()

    if not resume_res.data:
        raise HTTPException(status_code=404, detail="No master resume found.")

    # Get transcripts
    transcripts_res = supabase.table("transcripts")\
        .select("raw_text")\
        .eq("user_id", user_id)\
        .execute()

    transcript_texts = [t["raw_text"] for t in (transcripts_res.data or []) if t.get("raw_text")]

    # Generate resume via Claude
    resume_json = generate_resume(
        resume_text=resume_res.data["raw_text"],
        job_description=body.job_description,
        match_data=body.match_data,
        selected_improvements=body.selected_improvements,
        transcripts=transcript_texts
    )

    # Generate DOCX
    docx_bytes = generate_docx(resume_json)

    # Record application in database
    role = body.match_data.get("role", "Unknown Role")
    company = body.match_data.get("company")
    ats_score = resume_json.get("ats_score", 0)
    match_score = body.match_data.get("overall", 0)

    candidate_name = (resume_json.get("name") or "Resume").replace(" ", "_")
    role_slug = re.sub(r"[^a-zA-Z0-9]", "_", role)[:30]
    file_name = f"{candidate_name}_{role_slug}.docx"

    supabase.table("applications").insert({
        "user_id": user_id,
        "resume_id": resume_res.data["id"],
        "job_title": role,
        "company": company,
        "job_description": body.job_description[:500],
        "match_score": match_score,
        "ats_score": ats_score,
        "matched_keywords": body.match_data.get("matched", []),
        "missing_keywords": body.match_data.get("missing", []),
        "output_file_name": file_name,
        "improvements_applied": len(body.selected_improvements),
        "status": "complete",
    }).execute()

    # Increment monthly resume counter
    supabase.table("profiles")\
        .update({"resumes_used_this_month": profile.get("resumes_used_this_month", 0) + 1})\
        .eq("id", user_id)\
        .execute()

    # Return DOCX file
    return Response(
        content=docx_bytes,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": f'attachment; filename="{file_name}"'}
    )


# ── Get user files ────────────────────────────────────────────
@app.get("/api/files")
async def get_files(ctx: dict = Depends(get_current_user)):
    supabase = ctx["supabase"]
    user_id = ctx["profile"]["id"]

    resume_res = supabase.table("resumes")\
        .select("id, file_name, file_size, created_at")\
        .eq("user_id", user_id)\
        .eq("is_master", True)\
        .execute()

    transcripts_res = supabase.table("transcripts")\
        .select("id, file_name, created_at")\
        .eq("user_id", user_id)\
        .execute()

    return {
        "resume": resume_res.data[0] if resume_res.data else None,
        "transcripts": transcripts_res.data or [],
    }
