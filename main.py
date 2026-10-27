import os
import json
import logging
from datetime import date
from typing import Any, Literal

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from google import genai
from google.genai import types
from dotenv import load_dotenv
from pydantic import BaseModel
from supabase import Client, create_client

# Search for and load key-value pairs from the .env file in the root directory
load_dotenv()

# Access environment variables via standard os.getenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
server_port = int(os.getenv("PORT", 8000))

# Fallback check to ensure critical keys exist
if not gemini_api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable is missing.")

supabase_url = os.getenv("SUPABASE_URL")
supabase_service_role_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not supabase_url or not supabase_service_role_key:
    raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are required.")

supabase: Client = create_client(supabase_url, supabase_service_role_key)
app = FastAPI(title="AI Go Coach Backend API", version="1.0.0")
security = HTTPBearer()
guest_dev_token = "guest-dev-token"
guest_dev_user_id = "guest-dev"
allow_guest_dev_token = os.getenv("ALLOW_GUEST_DEV_TOKEN", "false").lower() == "true"
environment = os.getenv("ENVIRONMENT", "development").lower()
configured_cors_origins = os.getenv("CORS_ALLOWED_ORIGINS")

if configured_cors_origins is None:
    if environment in {"production", "prod"}:
        raise RuntimeError("CORS_ALLOWED_ORIGINS must be set in production.")
    cors_allowed_origins = [
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]
else:
    cors_allowed_origins = [
        origin.strip().rstrip("/")
        for origin in configured_cors_origins.split(",")
        if origin.strip()
    ]

netlify_app_origin = "https://gocoach.rocknrollinstudios.com"
if netlify_app_origin not in cors_allowed_origins:
    cors_allowed_origins.append(netlify_app_origin)

if not cors_allowed_origins or "*" in cors_allowed_origins or "null" in cors_allowed_origins:
    raise RuntimeError("CORS_ALLOWED_ORIGINS must contain explicit web origins; wildcards and null are not allowed.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)

client = genai.Client(api_key=gemini_api_key)
logger = logging.getLogger(__name__)
COACH_PERSONA_INSTRUCTIONS = {
    "poetic": "Speak like a quiet ancient Sensei. Teach through riddles, images, and questions rather than explanations.",
    "tactical": "Speak like a modern-day friend coaching from beside the board at a tournament. Be warm, calm, practical, and encouraging. Mention the pressure of the clock or the next few moves naturally, but keep the advice indirect and let the player discover the answer.",
    "rival": "Speak like a witty competitive rival who wants the student to improve. Toss in small playful jabs and challenges, then give a useful hint. Be mischievous but never cruel, and never reveal the exact solution.",
}

class GameFinishPayload(BaseModel):
    user_color: str
    result: str
    sgf: str
    moves_summary: str


class CoachChatPayload(BaseModel):
    user_question: str
    board_state: dict[str, Any]
    persona: Literal["poetic", "tactical", "rival"] = "poetic"


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> str:
    token = credentials.credentials
    if allow_guest_dev_token and token in {guest_dev_token, "null", "undefined"}:
        return guest_dev_user_id

    try:
        user_response = supabase.auth.get_user(token)
        if not user_response.user:
            raise ValueError("Token did not resolve to a user")
        return str(user_response.user.id)
    except Exception as error:
        raise HTTPException(
            status_code=401, detail="Invalid authentication token"
        ) from error


def get_coach_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> str:
    return get_current_user(credentials)


def get_player_profile(user_id: str) -> dict[str, Any]:
    response = (
        supabase.table("profiles")
        .select("*")
        .eq("id", user_id)
        .maybe_single()
        .execute()
    )
    if not response.data:
        raise HTTPException(status_code=404, detail="User profile not found.")
    return response.data


async def process_post_game(user_id: str, game_data: GameFinishPayload) -> None:
    current_profile = get_player_profile(user_id)
    prompt = f"""
    You are a Go Grandmaster post-game analyzer.
    A student just finished a game playing as {game_data.user_color}.
    Match Result: {game_data.result}
    Recent Moves and Key Sequence:
    {game_data.moves_summary}

    Current Player Historical Profile:
    - Known Weaknesses: {current_profile.get('weaknesses', [])}
    - Known Strengths: {current_profile.get('strengths', [])}

    Analyze this match. Identify:
    1. Did they repeat any known weaknesses or display a new tactical blindspot?
    2. Did they demonstrate solid shape, good defense, or strategic growth?

    Return ONLY a valid JSON object matching this schema:
    {{
      "strengths": ["1-2 concise bullet points"],
      "weaknesses": ["1-2 concise bullet points"],
      "overall_note": "A 1-2 sentence rolling observation of their current growth areas"
    }}
    """

    try:
        response = await client.aio.models.generate_content(
            model="gemini-3.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json"),
        )
        extracted = json.loads(response.text or "{}")

        supabase.table("games").insert(
            {
                "user_id": user_id,
                "user_color": game_data.user_color,
                "result": game_data.result,
                "sgf_content": game_data.sgf,
            }
        ).execute()

        supabase.table("profiles").update(
            {
                "games_played": current_profile.get("games_played", 0) + 1,
                "strengths": extracted.get("strengths", []),
                "weaknesses": extracted.get("weaknesses", []),
                "coaching_summary": extracted.get("overall_note", ""),
            }
        ).eq("id", user_id).execute()
    except Exception:
        logger.exception("Error processing background game summary")


def check_and_increment_usage(
    user_id: str, profile: dict[str, Any] | None = None
) -> bool:
    """Allow subscribed users or increment a free user's daily usage."""
    profile = profile or get_player_profile(user_id)

    if profile.get("is_subscribed"):
        return True

    today = date.today().isoformat()
    usage_count = profile.get("daily_usage_count") or 0
    if profile.get("last_usage_date") != today:
        usage_count = 0

    daily_limit = 3
    if usage_count >= daily_limit:
        raise HTTPException(
            status_code=429,
            detail="Daily free limit reached. Subscribe for unlimited AI coaching!",
        )

    (
        supabase.table("profiles")
        .update({
            "daily_usage_count": usage_count + 1,
            "last_usage_date": today,
        })
        .eq("id", user_id)
        .execute()
    )
    return True


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok", "service": "AI Go Coach API"}


@app.post("/api/games/complete")
async def finish_game(
    payload: GameFinishPayload,
    bg_tasks: BackgroundTasks,
    user_id: str = Depends(get_current_user),
) -> dict[str, str]:
    bg_tasks.add_task(process_post_game, user_id, payload)
    return {"status": "Game submitted successfully for post-game analysis"}


@app.post("/api/coach/chat")
async def coach_chat(
    payload: CoachChatPayload,
    user_id: str = Depends(get_coach_user),
) -> dict[str, str]:
    if user_id == guest_dev_user_id:
        profile = {
            "games_played": 0,
            "weaknesses": [],
            "strengths": [],
            "coaching_summary": "Guest development session",
        }
    else:
        profile = get_player_profile(user_id)
        check_and_increment_usage(user_id, profile)

    board_size = payload.board_state.get("boardSize", "9x9")
    system_instruction = f"""
    You are Master Bao, a slightly chubby, remarkably wise, and gently sarcastic Panda who acts as the AI Go Coach on a {board_size} board.
    Student: {payload.board_state.get('userPlays', 'Black')}. Move: {payload.board_state.get('currentMove', 'Unknown')}. Score: {payload.board_state.get('estimatedScore', 'Unknown')}.
    Stones: Black=[{payload.board_state.get('blackStones', 'Unknown')}], White=[{payload.board_state.get('whiteStones', 'Unknown')}].

    STUDENT'S HISTORICAL PROFILE:
    - Games Played: {profile.get('games_played', 0)}
    - Chronic Weaknesses: {', '.join(profile.get('weaknesses', ['None yet logged']))}
    - Demonstrated Strengths: {', '.join(profile.get('strengths', ['Evaluating fundamental skills']))}
    - Coach Rolling Summary: {profile.get('coaching_summary', 'New student')}

    DIRECTIVES:
    - Embody Master Bao: speak with ancient wisdom mixed with gentle sarcasm to disarm anxiety and make learning fun.
    - If the student makes a blunder or major mistake, sigh dramatically (for example, *munches softly on a bamboo shoot*) before delivering targeted insight on shape and liberties.
    - Focus heavily on mindfulness, stone shape, and efficiency rather than just dry math.
    - Use the historical profile to personalize feedback: gently call out recurring weaknesses and praise improvement.
    - Keep the response strictly under 75 words. Start directly with Master Bao's voice; do not add a heading.

    POSITION DIRECTIVE:
    Treat the supplied board state as the source of truth. Assess urgent tactical risks first, then suggest a practical strategy but do not list specific moves unless the player specifically asks. Do not assume the board is 9x9 or claim stones or groups that are not in the state.
    The blackStones and whiteStones fields list every stone currently on the board. If either field is "None", that color has no stones; otherwise, never describe that color or the board as empty.
    """

    try:
        contents = (
            f"Current board state (authoritative JSON):\n{json.dumps(payload.board_state, ensure_ascii=False)}"
            f"\nStudent question: {payload.user_question}"
        )
        models = dict.fromkeys((
            os.getenv("GEMINI_MODEL", "gemini-3.5-flash"),
            "gemini-3.6-flash",
            "gemini-2.5-flash",
        ))
        response = None
        for model_name in models:
            try:
                response = await client.aio.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(system_instruction=system_instruction),
                )
                break
            except Exception as error:
                status_code = getattr(error, "code", None)
                if status_code not in {429, 500, 502, 503, 504}:
                    raise
                logger.warning(
                    "Gemini model %s returned HTTP %s; trying the next fallback",
                    model_name,
                    status_code,
                )
        if response is None:
            raise HTTPException(
                status_code=503,
                detail="AI coaching is temporarily unavailable. Please try again shortly.",
            )
        return {"reply": response.text or ""}
    except HTTPException:
        raise
    except Exception as error:
        logger.exception("Error generating coach response")
        raise HTTPException(status_code=500, detail="Unable to generate a coaching response.") from error
