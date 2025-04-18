from typing import Annotated, Dict, List, Optional
import uuid
import random
from datetime import datetime
import hashlib

import aiohttp
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Path, Query, Request
from fastapi.security import HTTPBearer
from sqlmodel import Session, SQLModel, create_engine
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from datetime import date
from sqlalchemy.exc import IntegrityError
from sqlmodel import select
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from datetime import date
import random
from appwrite.client import Client
from appwrite.services.account import Account
from appwrite.services.users import Users


from models import *
from prompt_task import assign_prompt_owners
from utils import get_env

postgres_url = f'postgresql+psycopg://{get_env("DB_USER")}:{get_env("DB_PWD")}@{get_env("DB_HOST")}:{get_env("DB_PORT")}/{get_env("DB_NAME")}'

engine = create_engine(postgres_url, echo=True)
load_dotenv()


def create_db_and_tables():
    SQLModel.metadata.create_all(engine)


def get_session():
    with Session(engine) as session:
        yield session


SessionDep = Annotated[Session, Depends(get_session)]

app = FastAPI(
    on_startup=[
        create_db_and_tables,
    ],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
router = APIRouter(
    prefix="/api/v1",
    responses={404: {"description": "Not found"}},
)


def cleanup_expired_invite_codes():
    """Deletes expired invite codes from the database."""
    with Session(engine) as session:
        now = datetime.utcnow()
        statement = select(BubbleInviteCode).where(BubbleInviteCode.expires_at < now)
        expired_codes = session.exec(statement).all()
        if expired_codes:
            print(f"Deleting {len(expired_codes)} expired invite codes...")
            for code in expired_codes:
                session.delete(code)
            session.commit()
        else:
            print("No expired invite codes found.")


scheduler = BackgroundScheduler()
scheduler.add_job(
    assign_prompt_owners,
    CronTrigger(hour=0, minute=5),
    args=[engine, scheduler],  # pass scheduler instance to the task
    id="assign_prompt_owners_job",  # job ID
    replace_existing=True,  # overwrite the job if it exists
)
# clean up invide codes hourly
scheduler.add_job(
    cleanup_expired_invite_codes,
    CronTrigger(hour="*"),
    id="cleanup_invites_job",  # job ID
    replace_existing=True,  # replace the job if it exists because we might be running multiple instances
)
scheduler.start()


security = HTTPBearer()


async def authenticate(
    req: Request,
    session: SessionDep,
) -> UserContext:
    """Authenticate users with their JWT from Appwrite"""
    try:
        authorization = req.headers.get("Authorization", None)
        if not authorization:
            raise HTTPException(
                status_code=401,
                detail="Unauthorized; an Authorization header is necessary.",
            )
        if "Bearer" in authorization:
            token = authorization.split(" ")[1]
        else:
            token = authorization

        authClient = Client()
        authClient.set_endpoint(get_env("APPWRITE_ENDPOINT"))
        authClient.set_project(get_env("APPWRITE_PROJECT_ID"))
        authClient.set_jwt(token)
        account = Account(authClient)
        user_data = account.get()
        req.user_id = user_data["$id"]

        id = user_data["$id"]
        email = user_data["email"]

        # --- Mock User for Testing ---
        # id = "*************"
        # email = "**********"
        # --- End Mock User ---

        # check if user exists in the database
        statement = select(User).where(User.id == id)
        db_user = session.exec(statement).first()
        if not db_user:
            # create user if it doesn't exist
            db_user = User(
                id=id,
                email=email,
                display_name=email.split("@")[0],
                profile_picture=None,
            )
            session.add(db_user)
            session.commit()
            session.refresh(db_user)

        return UserContext(
            id=id,
            email=email,
            jwt=token,
        )

    except Exception as e:
        print(f"Authentication error: {e}")
        raise HTTPException(status_code=401, detail="Unauthorized; invalid token.")


async def get_user(
    user_id: str,
    session: SessionDep,
):
    """Get user from the database."""
    user = session.exec(select(User).where(User.id == user_id)).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


# --- Bubble Endpoints ---


@router.post(
    "/bubbles",
    response_model=Bubble,
    tags=["Bubbles"],
    summary="Create a new bubble",
    description="Create a new bubble. The user will be automatically added as a member.",
    dependencies=[Depends(security)],
)
async def create_bubble(
    bubble_data: BubbleCreate,
    session: SessionDep,
    user=Depends(authenticate),
):
    bubble = Bubble(
        name=bubble_data.name,
        id=str(uuid.uuid4()),
    )
    session.add(bubble)
    session.flush()  # so that bubble.id is populated

    session.add(
        BubbleMember(
            bubble_id=bubble.id,
            user_id=user.id,
        )
    )
    session.commit()
    session.refresh(bubble)
    return bubble


@router.post(
    "/bubbles/{bubble_id}/invite-code",
    tags=["Bubbles"],
    summary="Generate an invite code for a bubble",
    description="Generates a temporary 8-digit code to invite others to this bubble. The code expires in 1 day. Only members can generate codes.",
    response_model=Dict[str, str],
    dependencies=[Depends(security)],
)
async def generate_invite_code(
    bubble_id: str,
    session: SessionDep,
    user=Depends(authenticate),
):
    member = session.exec(
        select(BubbleMember).where(
            BubbleMember.bubble_id == bubble_id, BubbleMember.user_id == user.id
        )
    ).first()
    if not member:
        raise HTTPException(
            status_code=403, detail="Only members can generate invite codes."
        )

    # generate code
    while True:
        code = "".join(random.choices("0123456789", k=8))
        # is code active?
        existing_code = session.exec(
            select(BubbleInviteCode).where(
                BubbleInviteCode.code == code,
                BubbleInviteCode.expires_at > datetime.utcnow(),
            )
        ).first()
        if not existing_code:
            break  # unique code found

    invite = BubbleInviteCode(
        bubble_id=bubble_id,
        code=code,
        created_by_user_id=user.id,
    )
    session.add(invite)
    session.commit()
    session.refresh(invite)

    return {"invite_code": invite.code, "expires_at": str(invite.expires_at)}


@router.post(
    "/bubbles/join/{invite_code}",
    tags=["Bubbles"],
    summary="Join a bubble using an invite code",
    description="Join a bubble using a temporary 8-digit invite code.",
    dependencies=[Depends(security)],
)
async def join_bubble_with_code(
    session: SessionDep,
    invite_code: str = Path(..., description="The 8-digit invite code"),
    user=Depends(authenticate),
):
    # Find the invite code
    statement = select(BubbleInviteCode).where(
        BubbleInviteCode.code == invite_code,  # matches
        BubbleInviteCode.expires_at > datetime.utcnow(),  # not expired
    )
    invite = session.exec(statement).first()

    if not invite:
        raise HTTPException(status_code=404, detail="Invalid or expired invite code.")

    bubble_id = invite.bubble_id

    existing_member = session.exec(
        select(BubbleMember).where(
            BubbleMember.bubble_id == bubble_id, BubbleMember.user_id == user.id
        )
    ).first()
    if existing_member:
        raise HTTPException(status_code=400, detail="Already a member of this bubble.")

    try:
        session.add(BubbleMember(bubble_id=bubble_id, user_id=user.id))
        # delete the invite code after successful use
        session.delete(invite)
        session.commit()
    except IntegrityError:  # shouldn't happen
        session.rollback()
        raise HTTPException(status_code=500, detail="Failed to join bubble.")

    bubble = session.get(Bubble, bubble_id)
    return {
        "message": f"Successfully joined bubble '{bubble.name if bubble else 'Unknown'}'"
    }


@router.post(
    "/bubbles/{bubble_id}/leave",
    tags=["Bubbles"],
    summary="Leave a bubble",
    description="Leave a bubble by its ID (UUID).",
    dependencies=[Depends(security)],
)
async def leave_bubble(
    bubble_id: str,
    session: SessionDep,
    user=Depends(authenticate),
):
    bubble = session.get(Bubble, bubble_id)
    if not bubble:
        raise HTTPException(status_code=404, detail="Bubble not found")

    statement = (
        select(BubbleMember)
        .where(BubbleMember.bubble_id == bubble_id)
        .where(BubbleMember.user_id == user.id)
    )
    membership = session.exec(statement).first()

    if not membership:
        raise HTTPException(status_code=400, detail="Not a member of this bubble")

    session.delete(membership)
    session.commit()

    # check if bubble should be deleted (only if no members left)
    remaining_members_count = session.exec(
        select(BubbleMember).where(BubbleMember.bubble_id == bubble_id)
    ).first()

    if not remaining_members_count:
        # also delete pending invite codes for this bubble
        invite_codes = session.exec(
            select(BubbleInviteCode).where(BubbleInviteCode.bubble_id == bubble_id)
        ).all()
        for code in invite_codes:
            session.delete(code)

        session.delete(bubble)
        session.commit()
        return {
            "message": f"Left and deleted bubble '{bubble.name}' as you were the only member."
        }

    return {"message": f"Left bubble '{bubble.name}'"}


@router.get(
    "/bubbles",
    response_model=List[Bubble],
    tags=["Bubbles"],
    summary="Get user's bubbles",
    description="Get all bubbles the user is a member of.",
    dependencies=[Depends(security)],
)
async def get_user_bubbles(
    session: SessionDep,
    user=Depends(authenticate),
):
    statement = select(Bubble).join(BubbleMember).where(BubbleMember.user_id == user.id)
    bubbles = session.exec(statement).all()
    return bubbles


@router.post(
    "/bubbles/{bubble_id}/answer",
    tags=["Bubbles"],
    summary="Submit an answer to today's prompt",
    description="Submit an answer to today's prompt for a bubble. The user must be a member of the bubble.",
    dependencies=[Depends(security)],
)
async def submit_prompt_answer(
    session: SessionDep,
    bubble_id: str,
    answer: str = Query(..., min_length=1, max_length=1000),
    user=Depends(authenticate),
):
    today = date.today()

    membership = session.exec(
        select(BubbleMember).where(
            BubbleMember.bubble_id == bubble_id,
            BubbleMember.user_id == user.id,
        )
    ).first()

    if not membership:
        raise HTTPException(status_code=403, detail="You're not in this bubble")

    existing_answer = session.exec(
        select(DailyAnswer).where(
            DailyAnswer.bubble_id == bubble_id,
            DailyAnswer.user_id == user.id,
            DailyAnswer.date == today,
        )
    ).first()

    if existing_answer:
        existing_answer.answer = answer
        session.add(existing_answer)
    else:
        new_answer = DailyAnswer(
            bubble_id=bubble_id,
            user_id=user.id,
            date=today,
            answer=answer,
        )
        session.add(new_answer)

    session.commit()
    return {"message": "Answer submitted"}


@router.get(
    "/bubbles/{bubble_id}/answers",
    tags=["Bubbles"],
    summary="Get today's answers for a bubble",
    description="Get the responses users have submitted for a bubble's daily prompt. The user must be a member of the bubble.",
    response_model=List[Dict[str, Optional[str]]],
    dependencies=[Depends(security)],
)
async def get_answers_for_today(
    bubble_id: str,
    session: SessionDep,
    user=Depends(authenticate),
):
    member = session.exec(
        select(BubbleMember).where(
            BubbleMember.bubble_id == bubble_id, BubbleMember.user_id == user.id
        )
    ).first()

    if not member:
        raise HTTPException(status_code=403, detail="You're not in this bubble")

    today = date.today()

    statement = (
        select(DailyAnswer, User)
        .join(User, DailyAnswer.user_id == User.id)
        .where(DailyAnswer.bubble_id == bubble_id, DailyAnswer.date == today)
    )

    results = session.exec(statement).all()
    return [
        {
            "user_id": db_user.id,
            "display_name": db_user.display_name,
            "profile_picture": db_user.profile_picture,
            "answer": answer.answer,
        }
        for answer, db_user in results
    ]


@router.post(
    "/bubbles/{bubble_id}/prompt",
    tags=["Bubbles"],
    summary="Submit a prompt for today",
    description="Submit a prompt in response to a bubble's daily prompt. The user must be a member of the bubble.",
    dependencies=[Depends(security)],
)
async def submit_prompt(
    session: SessionDep,
    bubble_id: str,
    prompt: str = Query(..., min_length=1, max_length=200),
    user=Depends(authenticate),
):
    today = date.today()

    member = session.exec(
        select(BubbleMember).where(
            BubbleMember.bubble_id == bubble_id, BubbleMember.user_id == user.id
        )
    ).first()

    if not member:
        raise HTTPException(status_code=403, detail="Not a member")

    # is there already a prompt for today?
    existing = session.exec(
        select(DailyPrompt).where(
            DailyPrompt.bubble_id == bubble_id, DailyPrompt.date == today
        )
    ).first()

    if existing:
        # if so, update it
        existing.prompt = prompt
        existing.chosen_by_user_id = user.id
        session.add(existing)  # add to session for update
    else:
        new_prompt = DailyPrompt(
            bubble_id=bubble_id,
            date=today,
            prompt=prompt,
            chosen_by_user_id=user.id,
        )
        session.add(new_prompt)
    session.commit()
    return {"message": "Prompt submitted"}


@router.get(
    "/bubbles/{bubble_id}/prompt",
    tags=["Bubbles"],
    summary="Get today's prompt",
    description="Get today's prompt for a bubble (the user must be a member of the bubble)",
    response_model=Dict[str, Optional[str]],
    dependencies=[Depends(security)],
)
async def get_today_prompt(
    session: SessionDep,
    bubble_id: str = Path(...),
    user=Depends(authenticate),
):
    """Get today's prompt for a bubble."""
    # check user is a member
    member = session.exec(
        select(BubbleMember).where(
            BubbleMember.bubble_id == bubble_id, BubbleMember.user_id == user.id
        )
    ).first()
    if not member:
        raise HTTPException(status_code=403, detail="Not a member of this bubble")

    today = date.today()
    prompt = session.exec(
        select(DailyPrompt).where(
            DailyPrompt.bubble_id == bubble_id, DailyPrompt.date == today
        )
    ).first()

    if not prompt:
        return {
            "prompt": None,
            "chosen_by_user_id": None,
            "date": str(today),
        }

    return {
        "prompt": prompt.prompt,
        "chosen_by_user_id": prompt.chosen_by_user_id,
        "date": str(prompt.date),
    }


@router.get(
    "/bubbles/{bubble_id}/prompt-assignment",
    tags=["Bubbles"],
    summary="Get today's prompt assignment",
    description="Get information about who is assigned to create today's prompt and if they've done it already.",
    dependencies=[Depends(security)],
)
async def get_prompt_assignment(
    session: SessionDep,
    bubble_id: str = Path(...),
    user=Depends(authenticate),
):
    """Get information about who is assigned to create today's prompt and if they've done it."""
    # user must be a member of the bubble
    member = session.exec(
        select(BubbleMember).where(
            BubbleMember.bubble_id == bubble_id, BubbleMember.user_id == user.id
        )
    ).first()
    if not member:
        raise HTTPException(status_code=403, detail="Not a member of this bubble")

    today = date.today()

    # does today's prompt already exist?
    prompt = session.exec(
        select(DailyPrompt).where(
            DailyPrompt.bubble_id == bubble_id, DailyPrompt.date == today
        )
    ).first()

    if prompt:
        # prompt already written
        user_info = session.exec(
            select(User).where(User.id == prompt.chosen_by_user_id)
        ).first()

        return {
            "assigned_user_id": prompt.chosen_by_user_id,
            "display_name": user_info.display_name if user_info else "Unknown User",
            "profile_picture": user_info.profile_picture if user_info else None,
            "prompt_submitted": True,
            "prompt": prompt.prompt,
            "date": str(prompt.date),
        }

    members = session.exec(
        select(BubbleMember).where(BubbleMember.bubble_id == bubble_id)
    ).all()

    if not members:
        # shouldn't happen if the requesting user is a member, but handle anyway :D
        return {
            "assigned_user_id": None,
            "display_name": None,
            "profile_picture": None,
            "prompt_submitted": False,
            "date": str(today),
        }

    seed = f"{bubble_id}-{today}"
    hasher = hashlib.sha256(seed.encode())  # Use hashlib so it's consistent
    hash_int = int(hasher.hexdigest(), 16)
    chosen_member_index = hash_int % len(members)
    chosen_member = members[chosen_member_index]

    user_info = session.exec(
        select(User).where(User.id == chosen_member.user_id)
    ).first()

    return {
        "assigned_user_id": chosen_member.user_id,
        "display_name": user_info.display_name if user_info else "Unknown User",
        "profile_picture": user_info.profile_picture if user_info else None,
        "prompt_submitted": False,
        "date": str(today),
    }


# --- User Endpoints ---


@router.post(
    "/user/name",
    tags=["User"],
    summary="Set the user's display name",
    description="Set the user's display name. This is used to set the user's display name in the database and appwrite instance.",
    dependencies=[Depends(security)],
)
async def set_user_name(
    session: SessionDep,
    name: str = Query(..., min_length=1, max_length=100),
    user=Depends(authenticate),
):
    """Set the user's display name."""
    db_user = session.exec(
        select(User).where(User.id == user.id)
    ).first()  # Renamed variable
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")

    db_user.display_name = name
    session.add(db_user)  # Add to session for update
    session.commit()
    session.refresh(db_user)

    # --- Appwrite Update (legacy) ---
    async with aiohttp.ClientSession() as httpsession:
        try:
            http_res = await httpsession.patch(
                f"{get_env('APPWRITE_ENDPOINT')}/users/{db_user.id}/name",
                headers={
                    "x-appwrite-project": get_env("APPWRITE_PROJECT_ID"),
                    "x-appwrite-key": get_env("APPWRITE_API_KEY"),
                    "user-agent": "ApppwritePythonSDK/7.0.0",
                    "x-sdk-name": "Python",
                    "x-sdk-platform": "server",
                    "x-sdk-language": "python",
                    "x-sdk-version": "7.0.0",
                    "content-type": "application/json",
                },
                json={"name": name},
            )
            if http_res.status not in [200, 201]:
                res = await http_res.json()
                print(res)
                raise HTTPException(
                    status_code=http_res.status,
                    detail="Failed to update Appwrite user",
                )

        except Exception as e:
            raise HTTPException(
                status_code=500, detail="Failed to update Appwrite user"
            )
    # --- End Appwrite Update ---

    return {"message": "Display name updated"}


@router.post(
    "/user/pfp",
    tags=["User"],
    summary="Set the user's profile picture",
    description="Set the user's profile picture, the url should usually be a link to a file in an appwrite storage bucket",
    dependencies=[Depends(security)],
)
async def set_user_pfp(
    session: SessionDep,
    pfp_url: str = Query(..., min_length=1, max_length=1000),
    user=Depends(authenticate),
):
    """Set the user's profile picture."""
    db_user = session.exec(select(User).where(User.id == user.id)).first()
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")

    if pfp_url.lower() == "none":
        db_user.profile_picture = None
    else:
        db_user.profile_picture = pfp_url

    session.add(db_user)
    session.commit()
    session.refresh(db_user)

    return {"message": "Profile picture updated"}


@router.get(
    "/user/pfp/{user_id}",
    tags=["User"],
    summary="Returns the user's profile picture",
    response_model=Optional[str],
)
async def get_user_pfp(
    session: SessionDep,
    user_id=Path(..., title="The ID of the user to get the profile picture for"),
):
    """Get the user's profile picture."""
    db_user = session.exec(select(User).where(User.id == user_id)).first()
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")

    return db_user.profile_picture


@router.get(
    "/user/name/{user_id}",
    tags=["User"],
    summary="Get the user's display name",
    response_model=Dict[str, str],
    description="Get the user's display name by their ID.",
)
async def get_user_name(
    session: SessionDep,
    # user=Depends(authenticate), # auth not really needed to view name
    user_id=Path(..., title="The ID of the user to get the display name for"),
):
    """Get the user's display name."""
    db_user = session.exec(select(User).where(User.id == user_id)).first()
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")

    return {
        "name": db_user.display_name,
    }


@router.get(
    "/bubbles/{bubble_id}/members",
    tags=["Bubbles"],
    summary="Get all members of a bubble",
    description="Get all members of a bubble by its ID. Requires authentication.",
    response_model=List[User],
    dependencies=[Depends(security)],
)
async def get_bubble_members(
    bubble_id: str,
    session: SessionDep,
    user=Depends(authenticate),
):
    """Get all members of a bubble."""
    member = session.exec(
        select(BubbleMember).where(
            BubbleMember.bubble_id == bubble_id, BubbleMember.user_id == user.id
        )
    ).first()
    if not member:
        raise HTTPException(
            status_code=403, detail="You must be a member to view other members."
        )

    bubble = session.get(Bubble, bubble_id)
    if not bubble:
        raise HTTPException(status_code=404, detail="Bubble not found")

    members = session.exec(
        select(User).join(BubbleMember).where(BubbleMember.bubble_id == bubble_id)
    ).all()

    return members


@router.get(
    "/user/feed",
    tags=["User"],
    summary="Get the user's activity feed",
    description="Returns a chronological list of events from the user's bubbles over the last 7 days.",
    response_model=List[FeedItem],
    dependencies=[Depends(security)],
)
async def get_user_feed(
    session: SessionDep,
    user=Depends(authenticate),
):
    """Generates an activity feed for the authenticated user."""
    # 1. get the user's bubbles
    user_bubble_memberships = session.exec(
        select(BubbleMember.bubble_id).where(BubbleMember.user_id == user.id)
    ).all()
    user_bubble_ids = [m for m in user_bubble_memberships]

    if not user_bubble_ids:
        return []  # N=no bubbles = no feed!

    # 2. last 7 days
    end_datetime = datetime.utcnow()
    start_datetime = end_datetime - timedelta(days=7)
    start_date = start_datetime.date()

    feed_items = []

    # 3. get  prompts submitted in the period
    prompt_results = session.exec(
        select(DailyPrompt, Bubble, User)
        .join(Bubble, DailyPrompt.bubble_id == Bubble.id)
        .join(User, DailyPrompt.chosen_by_user_id == User.id)
        .where(DailyPrompt.bubble_id.in_(user_bubble_ids))
        .where(DailyPrompt.date >= start_date)  # Prompts are date-based
    ).all()

    for prompt, bubble, author_user in prompt_results:
        # date of the prompt at noon UTC as its timestamp should be good enough for now
        prompt_timestamp = datetime.combine(
            prompt.date, datetime.min.time().replace(hour=12)
        )
        # it *must* fall within the precise 7-day window
        if prompt_timestamp >= start_datetime:
            feed_items.append(
                FeedItem(
                    type="prompt_submitted",
                    timestamp=prompt_timestamp,
                    bubble=FeedItemBubble(id=bubble.id, name=bubble.name),
                    actor_user=FeedItemUser(
                        id=author_user.id,
                        display_name=author_user.display_name,
                        profile_picture=author_user.profile_picture,
                    ),
                    content=prompt.prompt,
                )
            )

    # 4. get all answers submitted in the period
    answer_results = session.exec(
        select(DailyAnswer, Bubble, User)
        .join(Bubble, DailyAnswer.bubble_id == Bubble.id)
        .join(User, DailyAnswer.user_id == User.id)
        .where(DailyAnswer.bubble_id.in_(user_bubble_ids))
        .where(
            DailyAnswer.created_at >= start_datetime
        )  # Answers have precise timesttamps
    ).all()

    for answer, bubble, author_user in answer_results:
        feed_items.append(
            FeedItem(
                type="answer_submitted",
                timestamp=answer.created_at,
                bubble=FeedItemBubble(id=bubble.id, name=bubble.name),
                actor_user=FeedItemUser(
                    id=author_user.id,
                    display_name=author_user.display_name,
                    profile_picture=author_user.profile_picture,
                ),
                content=answer.answer,
            )
        )

    # 5. find users joining bubbles in the period
    join_results = session.exec(
        select(BubbleMember, Bubble, User)
        .join(Bubble, BubbleMember.bubble_id == Bubble.id)
        .join(User, BubbleMember.user_id == User.id)
        .where(BubbleMember.bubble_id.in_(user_bubble_ids))
        .where(
            BubbleMember.joined_at >= start_datetime
        )  # joins have precise timestamps. fancy!
    ).all()

    for member, bubble, joined_user in join_results:
        feed_items.append(
            FeedItem(
                type="user_joined",
                timestamp=member.joined_at,
                bubble=FeedItemBubble(id=bubble.id, name=bubble.name),
                target_user=FeedItemUser(
                    id=joined_user.id,
                    display_name=joined_user.display_name,
                    profile_picture=joined_user.profile_picture,
                ),
            )
        )

    # 6. sort chronologically
    feed_items.sort(key=lambda item: item.timestamp, reverse=True)

    return feed_items


@router.delete(
    "/user/close_account",
    tags=["User"],
    summary="Close user account",
    description="Close the user account and delete all data associated with it.",
    dependencies=[Depends(security)],
)
async def close_account(
    session: SessionDep,
    user=Depends(authenticate),
):
    """Close the user account and delete all data associated with it."""
    # delete all user data
    statement = select(User).where(User.id == user.id)
    db_user = session.exec(statement).first()
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")

    # delete all bubbles the user is a member of
    bubbles = session.exec(
        select(BubbleMember).where(BubbleMember.user_id == user.id)
    ).all()
    for bubble in bubbles:
        session.delete(bubble)

    # delete the user
    session.delete(db_user)
    session.commit()

    # delete the user from Appwrite
    adminClient = Client()
    adminClient.set_endpoint(get_env("APPWRITE_ENDPOINT"))
    adminClient.set_project(get_env("APPWRITE_PROJECT_ID"))
    adminClient.set_key(get_env("APPWRITE_API_KEY"))
    users = Users(adminClient)
    try:
        users.delete(user.id)
    except Exception as e:
        print(f"Failed to delete user from Appwrite: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to delete user from Appwrite"
        )

    return {"message": "Account closed successfully."}


@router.post(
    "/user/notification-time",
    tags=["User"],
    summary="Set preferred notification time",
    description="Set the user's preferred hour (0-23 UTC) for receiving 'your turn to prompt' notifications. Send null to use the default time.",
    dependencies=[Depends(security)],
)
async def set_notification_time(
    session: SessionDep,
    pref_data: UserNotificationTime,
    user=Depends(authenticate),
):
    db_user = session.exec(select(User).where(User.id == user.id)).first()
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")

    db_user.preferred_notification_hour_utc = pref_data.hour_utc
    session.add(db_user)
    session.commit()
    session.refresh(db_user)
    return {"message": "Notification preference updated."}


@router.get(
    "/user/notification-time",
    tags=["User"],
    summary="Get preferred notification time",
    description="Get the user's preferred hour (0-23 UTC) for receiving 'your turn to prompt' notifications.",
    response_model=UserNotificationTime,
    dependencies=[Depends(security)],
)
async def get_notification_time(
    session: SessionDep,
    user=Depends(authenticate),
):
    db_user = session.exec(select(User).where(User.id == user.id)).first()
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")

    return UserNotificationTime(hour_utc=db_user.preferred_notification_hour_utc)


@router.get(
    "/",
    tags=["Health Check"],
    summary="Health check",
    description="Check if the server is running.",
)
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


app.include_router(router)
