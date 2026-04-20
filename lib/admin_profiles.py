"""
Admin candidate profile management.
Handles CRUD for candidate profiles, their resumes, transcripts, and applications.
"""
from fastapi import HTTPException
from lib.text_extractor import extract_text


def require_admin(profile: dict):
    if profile.get("plan") != "admin":
        raise HTTPException(status_code=403, detail="Admin plan required.")


# ── Candidate Profiles ────────────────────────────────────────

def list_candidate_profiles(supabase, admin_id: str) -> list:
    res = supabase.table("candidate_profiles")\
        .select("*, candidate_resumes(id, file_name), candidate_transcripts(id, file_name)")\
        .eq("admin_id", admin_id)\
        .eq("is_active", True)\
        .order("created_at", desc=True)\
        .execute()
    return res.data or []


def create_candidate_profile(supabase, admin_id: str, full_name: str, email: str = None, notes: str = None) -> dict:
    if not full_name.strip():
        raise HTTPException(status_code=422, detail="Candidate name is required.")
    res = supabase.table("candidate_profiles").insert({
        "admin_id": admin_id,
        "full_name": full_name.strip(),
        "email": email,
        "notes": notes,
    }).execute()
    return res.data[0]


def delete_candidate_profile(supabase, admin_id: str, candidate_id: str):
    """
    Hard delete — removes profile and all associated data (resumes, transcripts, applications).
    Cascades via FK constraints.
    """
    res = supabase.table("candidate_profiles")\
        .delete()\
        .eq("id", candidate_id)\
        .eq("admin_id", admin_id)\
        .execute()
    if not res.data:
        raise HTTPException(status_code=404, detail="Candidate profile not found.")
    return {"message": "Candidate profile and all associated data deleted."}


# ── Candidate Resume ──────────────────────────────────────────

async def upload_candidate_resume(supabase, admin_id: str, candidate_id: str, file_bytes: bytes, filename: str) -> dict:
    # Verify candidate belongs to admin
    cp = supabase.table("candidate_profiles")\
        .select("id")\
        .eq("id", candidate_id)\
        .eq("admin_id", admin_id)\
        .single()\
        .execute()
    if not cp.data:
        raise HTTPException(status_code=404, detail="Candidate not found.")

    # Delete existing resume if any
    supabase.table("candidate_resumes")\
        .delete()\
        .eq("candidate_id", candidate_id)\
        .execute()

    if len(file_bytes) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large. Max 10MB.")

    raw_text = extract_text(file_bytes, filename)

    res = supabase.table("candidate_resumes").insert({
        "candidate_id": candidate_id,
        "admin_id": admin_id,
        "file_name": filename,
        "file_size": len(file_bytes),
        "raw_text": raw_text,
    }).execute()
    return res.data[0]


def delete_candidate_resume(supabase, admin_id: str, resume_id: str):
    res = supabase.table("candidate_resumes")\
        .delete()\
        .eq("id", resume_id)\
        .eq("admin_id", admin_id)\
        .execute()
    if not res.data:
        raise HTTPException(status_code=404, detail="Resume not found.")
    return {"message": "Resume deleted."}


# ── Candidate Transcripts ─────────────────────────────────────

async def upload_candidate_transcript(supabase, admin_id: str, candidate_id: str, file_bytes: bytes, filename: str) -> dict:
    cp = supabase.table("candidate_profiles")\
        .select("id")\
        .eq("id", candidate_id)\
        .eq("admin_id", admin_id)\
        .single()\
        .execute()
    if not cp.data:
        raise HTTPException(status_code=404, detail="Candidate not found.")

    if len(file_bytes) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large. Max 10MB.")

    if filename.lower().endswith(".txt"):
        raw_text = file_bytes.decode("utf-8", errors="ignore")
    else:
        raw_text = extract_text(file_bytes, filename)

    res = supabase.table("candidate_transcripts").insert({
        "candidate_id": candidate_id,
        "admin_id": admin_id,
        "file_name": filename,
        "raw_text": raw_text,
    }).execute()
    return res.data[0]


def delete_candidate_transcript(supabase, admin_id: str, transcript_id: str):
    res = supabase.table("candidate_transcripts")\
        .delete()\
        .eq("id", transcript_id)\
        .eq("admin_id", admin_id)\
        .execute()
    if not res.data:
        raise HTTPException(status_code=404, detail="Transcript not found.")
    return {"message": "Transcript deleted."}


# ── Candidate Files ───────────────────────────────────────────

def get_candidate_files(supabase, admin_id: str, candidate_id: str) -> dict:
    cp = supabase.table("candidate_profiles")\
        .select("id")\
        .eq("id", candidate_id)\
        .eq("admin_id", admin_id)\
        .single()\
        .execute()
    if not cp.data:
        raise HTTPException(status_code=404, detail="Candidate not found.")

    resume_res = supabase.table("candidate_resumes")\
        .select("id, file_name, file_size, created_at")\
        .eq("candidate_id", candidate_id)\
        .execute()

    transcripts_res = supabase.table("candidate_transcripts")\
        .select("id, file_name, created_at")\
        .eq("candidate_id", candidate_id)\
        .execute()

    return {
        "resume": resume_res.data[0] if resume_res.data else None,
        "transcripts": transcripts_res.data or [],
    }


# ── Candidate Applications ────────────────────────────────────

def save_candidate_application(supabase, admin_id: str, candidate_id: str,
                                job_title: str, company: str, job_description: str,
                                job_url: str, match_score: int, ats_score: int,
                                matched_keywords: list, missing_keywords: list,
                                output_file_name: str, output_file_data: bytes,
                                improvements_applied: int) -> dict:
    res = supabase.table("candidate_applications").insert({
        "candidate_id": candidate_id,
        "admin_id": admin_id,
        "job_title": job_title,
        "company": company,
        "job_description": job_description[:500],
        "job_url": job_url,
        "match_score": match_score,
        "ats_score": ats_score,
        "matched_keywords": matched_keywords,
        "missing_keywords": missing_keywords,
        "output_file_name": output_file_name,
        "output_file_data": output_file_data.hex(),  # store as hex string
        "improvements_applied": improvements_applied,
        "visible_to_customer": True,
        "status": "complete",
    }).execute()
    return res.data[0]


def list_candidate_applications(supabase, admin_id: str, candidate_id: str) -> list:
    res = supabase.table("candidate_applications")\
        .select("id, job_title, company, job_url, match_score, ats_score, output_file_name, improvements_applied, created_at")\
        .eq("candidate_id", candidate_id)\
        .eq("admin_id", admin_id)\
        .order("created_at", desc=True)\
        .execute()
    return res.data or []
