from fastapi import HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from supabase import create_client
import os

security = HTTPBearer()

def get_supabase():
    return create_client(
        os.environ["SUPABASE_URL"],
        os.environ["SUPABASE_SERVICE_ROLE_KEY"]
    )

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """Validate Supabase JWT and return user + profile."""
    token = credentials.credentials
    supabase = get_supabase()

    try:
        # Verify token with Supabase
        response = supabase.auth.get_user(token)
        user = response.user
        if not user:
            raise HTTPException(status_code=401, detail="Invalid token")
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    # Fetch profile with plan info
    profile_res = supabase.table("profiles").select("*").eq("id", user.id).single().execute()
    if not profile_res.data:
        raise HTTPException(status_code=404, detail="Profile not found")

    return {"user": user, "profile": profile_res.data, "supabase": supabase}
