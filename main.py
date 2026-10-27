import os
import json
import logging
from datetime import date
from typing import Any, Literal
from urllib.parse import urlsplit

import stripe
from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Request
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
stripe_secret_key = os.getenv("STRIPE_SECRET_KEY")
stripe_price_id = os.getenv("STRIPE_PRICE_ID")
stripe_webhook_secret = os.getenv("STRIPE_WEBHOOK_SECRET")
frontend_url = os.getenv("FRONTEND_URL")

if not supabase_url or not supabase_service_role_key:
    raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are required.")

supabase: Client = create_client(supabase_url, supabase_service_role_key)
app = FastAPI(title="AI Go Coach Backend API", version="1.0.0")
security = HTTPBearer()
guest_dev_token = "guest-dev-token"
guest_dev_user_id = "guest-dev"
allow_guest_dev_token = os.getenv("ALLOW_GUEST_DEV_TOKEN", "false").lower() == "true"
unlimited_test_user_id = os.getenv("UNLIMITED_TEST_USER_ID")
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
    allow_headers=["Authorization", "Content-Type", "X-User-ID"],
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
    game_stats: dict[str, Any] | None = None


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
    x_user_id: str | None = Header(default=None, alias="X-User-ID"),
) -> str:
    user_id = get_current_user(credentials)
    if x_user_id is not None and x_user_id != user_id:
        raise HTTPException(
            status_code=403,
            detail="X-User-ID does not match the authenticated user.",
        )
    return user_id


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
    try:
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

        response = await client.aio.models.generate_content(
            model="gemini-3.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json"),
        )
        extracted = json.loads(response.text or "{}")

        supabase.table("profiles").update(
            {
                "strengths": extracted.get("strengths", []),
                "weaknesses": extracted.get("weaknesses", []),
                "coaching_summary": extracted.get("overall_note", ""),
            }
        ).eq("id", user_id).execute()
    except Exception:
        logger.exception("Error processing background game summary")


@app.get("/api/profile/history")
def get_profile_history(
    user_id: str = Depends(get_current_user),
) -> dict[str, Any]:
    profile_response = (
        supabase.table("profiles")
        .select("*")
        .eq("id", user_id)
        .maybe_single()
        .execute()
    )
    games_response = (
        supabase.table("games")
        .select("user_color, result, sgf_content, game_stats")
        .eq("user_id", user_id)
        .execute()
    )
    profile = profile_response.data or {}
    return {
        "profile": {
            "games_played": profile.get("games_played", 0),
            "wins": profile.get("wins", 0),
            "losses": profile.get("losses", 0),
            "strengths": profile.get("strengths", []),
            "weaknesses": profile.get("weaknesses", []),
            "coaching_summary": profile.get("coaching_summary", ""),
            "is_subscribed": bool(profile.get("is_subscribed")),
        },
        "games": games_response.data or [],
    }


def get_configured_frontend_url() -> str:
    if not frontend_url:
        raise HTTPException(
            status_code=503,
            detail="Subscription checkout is not configured: FRONTEND_URL is missing.",
        )

    parsed_url = urlsplit(frontend_url)
    if (
        parsed_url.scheme not in {"http", "https"}
        or not parsed_url.netloc
        or parsed_url.username
        or parsed_url.password
        or parsed_url.path not in {"", "/"}
        or parsed_url.query
        or parsed_url.fragment
    ):
        raise HTTPException(
            status_code=503,
            detail="Subscription checkout is not configured with a valid FRONTEND_URL.",
        )

    return f"{parsed_url.scheme}://{parsed_url.netloc}"


@app.post("/api/billing/checkout-session")
def create_checkout_session(
    user_id: str = Depends(get_current_user),
) -> dict[str, str]:
    if user_id == guest_dev_user_id:
        raise HTTPException(status_code=403, detail="Log in before starting checkout.")
    if not stripe_secret_key or not stripe_price_id or not stripe_webhook_secret:
        raise HTTPException(
            status_code=503,
            detail="Subscription checkout and webhook are not fully configured on the server.",
        )

    profile = get_player_profile(user_id)
    if profile.get("is_subscribed"):
        raise HTTPException(status_code=409, detail="This account already has Pro access.")

    app_url = get_configured_frontend_url()
    stripe.api_key = stripe_secret_key
    try:
        session = stripe.checkout.Session.create(
            mode="subscription",
            line_items=[{"price": stripe_price_id, "quantity": 1}],
            client_reference_id=user_id,
            metadata={"supabase_user_id": user_id},
            subscription_data={"metadata": {"supabase_user_id": user_id}},
            success_url=f"{app_url}/?checkout=success&session_id={{CHECKOUT_SESSION_ID}}",
            cancel_url=f"{app_url}/?checkout=cancel",
        )
    except stripe.StripeError as error:
        logger.exception("Unable to create Stripe Checkout session")
        raise HTTPException(
            status_code=502,
            detail="Unable to start subscription checkout. Please try again.",
        ) from error

    if not session.url:
        logger.error("Stripe Checkout session %s did not include a URL", session.id)
        raise HTTPException(
            status_code=502,
            detail="Stripe did not return a checkout URL. Please try again.",
        )

    return {"url": session.url}


@app.post("/api/stripe-webhook")
@app.post("/api/billing/webhook")
async def handle_stripe_webhook(
    request: Request,
    stripe_signature: str = Header(default="", alias="Stripe-Signature"),
) -> dict[str, bool]:
    if not stripe_webhook_secret:
        raise HTTPException(
            status_code=503,
            detail="Subscription webhook is not configured on the server.",
        )

    try:
        event = stripe.Webhook.construct_event(
            payload=await request.body(),
            sig_header=stripe_signature,
            secret=stripe_webhook_secret,
        )
    except (ValueError, stripe.SignatureVerificationError) as error:
        raise HTTPException(
            status_code=400,
            detail="Invalid Stripe webhook payload or signature.",
        ) from error

    if isinstance(event, stripe.StripeObject):
        event = event.to_dict()

    event_type = event.get("type")
    data_object = event.get("data", {}).get("object", {})
    if event_type == "checkout.session.completed":
        if data_object.get("mode") != "subscription":
            return {"received": True}

        user_id = (
            data_object.get("client_reference_id")
            or (data_object.get("metadata") or {}).get("supabase_user_id")
        )
        customer_id = data_object.get("customer")
        if not user_id:
            logger.error(
                "Stripe Checkout event %s has no Supabase user reference",
                event.get("id"),
            )
            raise HTTPException(
                status_code=400,
                detail="Checkout event is missing its account reference.",
            )

        updates: dict[str, Any] = {}
        if customer_id:
            updates["stripe_customer_id"] = customer_id
        if data_object.get("payment_status") in {"paid", "no_payment_required"}:
            updates["is_subscribed"] = True
        if updates:
            update_response = (
                supabase.table("profiles")
                .update(updates)
                .eq("id", user_id)
                .select("id")
                .execute()
            )
            if not update_response.data:
                logger.error(
                    "Stripe Checkout event %s references missing profile %s",
                    event.get("id"),
                    user_id,
                )
                raise HTTPException(
                    status_code=500,
                    detail="The subscription could not be linked to the account.",
                )
        return {"received": True}

    if event_type not in {
        "customer.subscription.created",
        "customer.subscription.updated",
        "customer.subscription.deleted",
    }:
        return {"received": True}

    subscription = data_object
    metadata = subscription.get("metadata") or {}
    user_id = metadata.get("supabase_user_id")
    customer_id = subscription.get("customer")
    updates: dict[str, Any] = {
        "is_subscribed": (
            event_type != "customer.subscription.deleted"
            and subscription.get("status") in {"active", "trialing"}
        )
    }
    if customer_id:
        updates["stripe_customer_id"] = customer_id

    profile_query = supabase.table("profiles").update(updates)
    if user_id:
        profile_query = profile_query.eq("id", user_id)
    elif customer_id:
        profile_query = profile_query.eq("stripe_customer_id", customer_id)
    else:
        logger.warning(
            "Stripe subscription event %s has no Supabase user or Stripe customer reference",
            event.get("id"),
        )
        raise HTTPException(
            status_code=400,
            detail="Subscription event is missing its account reference.",
        )

    update_response = profile_query.select("id").execute()
    if not update_response.data:
        logger.error(
            "Stripe subscription event %s references no matching profile (user=%s, customer=%s)",
            event.get("id"),
            user_id,
            customer_id,
        )
        raise HTTPException(
            status_code=500,
            detail="The subscription could not be linked to the account.",
        )

    return {"received": True}


def check_and_increment_usage(
    user_id: str, profile: dict[str, Any] | None = None
) -> bool:
    """Allow subscribed users or increment a free user's daily usage."""
    if unlimited_test_user_id and user_id == unlimited_test_user_id:
        return True

    profile = profile or get_player_profile(user_id)

    if profile.get("is_subscribed"):
        return True

    today = date.today().isoformat()
    usage_count = profile.get("daily_usage_count") or 0
    if profile.get("last_usage_date") != today:
        usage_count = 0

    daily_limit = 5
    if usage_count >= daily_limit:
        raise HTTPException(
            status_code=429,
            detail="Daily free limit reached (5/5). Master Bao says: 'Even a panda needs rest—or a Pro subscription for infinite wisdom!'",
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
    winner_char = payload.result[:1].upper()
    user_char = payload.user_color[:1].upper()
    if winner_char not in {"B", "W"} or user_char not in {"B", "W"}:
        raise HTTPException(
            status_code=422,
            detail="Game result and user color must identify Black or White.",
        )

    profile = get_player_profile(user_id)
    user_won = winner_char == user_char
    supabase.table("games").insert(
        {
            "user_id": user_id,
            "user_color": payload.user_color,
            "result": payload.result,
            "sgf_content": payload.sgf,
            "game_stats": payload.game_stats or {},
        }
    ).execute()
    supabase.table("profiles").update(
        {
            "games_played": (profile.get("games_played") or 0) + 1,
            "wins": (profile.get("wins") or 0) + int(user_won),
            "losses": (profile.get("losses") or 0) + int(not user_won),
        }
    ).eq("id", user_id).execute()

    bg_tasks.add_task(process_post_game, user_id, payload)
    return {"status": "Game saved successfully; post-game analysis is processing."}


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
