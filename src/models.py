from datetime import datetime, date, timedelta  # Add timedelta
from typing import Optional, List
import uuid
from sqlmodel import Field, SQLModel, Relationship
from dataclasses import dataclass
from pydantic import BaseModel, conint


@dataclass
class UserContext:
    id: str
    email: str
    jwt: str


"""
User table - This is the main table for users. Each user can have many bubbles and many answers.
Users have a display name and a profile picture.
They can also have many memberships (bubbles they are part of) and many answers (answers they have given).
"""


# User table
class User(SQLModel, table=True):
    id: Optional[str] = Field(default=None, primary_key=True)
    email: str
    display_name: str
    profile_picture: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    # preferred notification hour (UTC, 0-23). None = default.
    preferred_notification_hour_utc: Optional[int] = Field(default=None, ge=0, le=23)

    memberships: List["BubbleMember"] = Relationship(back_populates="user")
    answers: List["DailyAnswer"] = Relationship(back_populates="user")
    created_invite_codes: List["BubbleInviteCode"] = Relationship(
        back_populates="created_by_user"
    )


"""
Bubble table - This is the main table for bubbles. A bubble is a group of users who can answer daily questions together and each bubble can have many members and many answers.
"""


class Bubble(SQLModel, table=True):
    id: Optional[str] = Field(
        primary_key=True,
        default_factory=lambda: str(uuid.uuid4()),
    )
    name: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

    members: List["BubbleMember"] = Relationship(back_populates="bubble")
    answers: List["DailyAnswer"] = Relationship(back_populates="bubble")
    invite_codes: List["BubbleInviteCode"] = Relationship(
        back_populates="bubble"
    )  # Add relationship


"""
BubbleMember table - This is a many-to-many relationship between users and bubbles.
"""


class BubbleMember(SQLModel, table=True):
    bubble_id: str = Field(foreign_key="bubble.id", primary_key=True)
    user_id: str = Field(foreign_key="user.id", primary_key=True)
    joined_at: datetime = Field(default_factory=datetime.utcnow)
    nickname: Optional[str] = None  # display name within bubble

    bubble: Bubble = Relationship(back_populates="members")
    user: User = Relationship(back_populates="memberships")


"""
DailyAnswer table - I use this to store each user in a bubble's answer to the daily question.
"""


class DailyAnswer(SQLModel, table=True):
    id: Optional[str] = Field(
        default_factory=lambda: str(uuid.uuid4()),
        primary_key=True,
    )
    bubble_id: str = Field(foreign_key="bubble.id", ondelete="CASCADE")
    user_id: str = Field(foreign_key="user.id", ondelete="CASCADE")
    date: date
    answer: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

    bubble: Bubble = Relationship(back_populates="answers")
    user: User = Relationship(back_populates="answers")


"""
DailyPrompt table - This is used to store the daily question for each bubble.
Each bubble can have a different question for each day.
"""


class DailyPrompt(SQLModel, table=True):
    id: Optional[str] = Field(
        default_factory=lambda: str(uuid.uuid4()),
        primary_key=True,
    )
    bubble_id: str = Field(foreign_key="bubble.id", ondelete="CASCADE")
    date: date
    prompt: str
    chosen_by_user_id: str = Field(foreign_key="user.id")


"""
BubbleInviteCode table - Stores temporary invite codes for bubbles.
"""


class BubbleInviteCode(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    bubble_id: str = Field(foreign_key="bubble.id", index=True)
    code: str = Field(index=True, unique=True)  # codes have to be unique!
    created_by_user_id: str = Field(foreign_key="user.id")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: datetime = Field(
        default_factory=lambda: datetime.utcnow() + timedelta(days=1)
    )  # expire in a day

    bubble: Bubble = Relationship(back_populates="invite_codes")
    created_by_user: User = Relationship(back_populates="created_invite_codes")


"""
Model for response when creating a new bubble
"""


class BubbleCreate(SQLModel):
    name: str = Field(min_length=1, max_length=100)


"""
Models for feed route
"""


class FeedItemUser(BaseModel):
    id: str
    display_name: Optional[str] = None
    profile_picture: Optional[str] = None


class FeedItemBubble(BaseModel):
    id: str
    name: Optional[str] = None


class FeedItem(BaseModel):
    type: str  # "prompt_submitted", "answer_submitted", "user_joined"
    timestamp: datetime
    bubble: FeedItemBubble
    actor_user: Optional[FeedItemUser] = None  # user who created prompt/answer
    target_user: Optional[FeedItemUser] = None  # user who joined
    content: Optional[str] = None  # prompt/answer text


# Model for notification settings
class UserNotificationTime(BaseModel):
    # Hour (UTC, 0-23) or None to reset
    hour_utc: Optional[int] = Field(default=None, ge=0, le=23)
