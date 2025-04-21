from datetime import date, datetime, time, timezone
import os

import hashlib
from sqlmodel import Session, select
from models import Bubble, BubbleMember, DailyPrompt, User
import onesignal
from onesignal.api import default_api
from onesignal.models import Notification
import dotenv
from apscheduler.schedulers.background import (
    BackgroundScheduler,
)


dotenv.load_dotenv()

osignal_config = onesignal.Configuration(
    app_key=os.getenv("ONESIGNAL_API_KEY"),
)

api_instance = default_api.DefaultApi(onesignal.ApiClient(osignal_config))


def sendNotification(
    message,
    userIds: list = [],
    title: str = "Notification",
    ttl: int = 60 * 10,
    filters: list = [],
    # channel: str = os.getenv("ONESIGNAL_GENERIC_CHANNEL"),
    priority: int = 10,
    small_icon="ic_stat_onesignal_default",
):
    """
    message: str = The message to send to the user
    userIds: list<str> = The user IDs to send the message to (these are the external user IDs from appwrite i.e. student IDs)
    ttl: int = The time to live for the notification in seconds, 10 minutes is reasonable for a bus notification
    headings: dict = The headings for the notification, this is optional and will default to the message if not provided
    """
    notification = Notification(
        app_id=os.environ.get("ONESIGNAL_APP_ID"),
        contents={"en": message},
        include_external_user_ids=userIds,
        ttl=ttl,
        headings={"en": title},
        filters=filters,
        # android_channel_id=channel,
        android_accent_color="737EFF",
        is_android=True,
        is_ios=True,
        priority=priority,
        small_icon=small_icon,
    )

    api_instance.create_notification(notification)


def assign_prompt_owners(engine, scheduler: BackgroundScheduler):
    """
    Assigns a user in each bubble to write the prompt for today if one hasn't been written.
    Uses a deterministic hash based on bubble ID and date to choose the user.
    Sends a notification to the chosen user.
    """
    with Session(engine) as session:
        today = date.today()
        now_utc = datetime.now(timezone.utc)
        default_notification_hour = (
            10  # default UTC hour is 10 AM; it's a sensible time
        )

        bubble_ids = session.exec(select(Bubble.id)).all()

        for bubble_id in bubble_ids:
            # check if prompt already exists
            existing_prompt = session.exec(
                select(DailyPrompt).where(
                    DailyPrompt.bubble_id == bubble_id, DailyPrompt.date == today
                )
            ).first()
            if existing_prompt:
                continue

            members = session.exec(
                select(BubbleMember).where(BubbleMember.bubble_id == bubble_id)
            ).all()
            if not members:
                continue

            # pick a member deterministically (fancy!)
            seed = f"{bubble_id}-{today}"
            hasher = hashlib.sha256(seed.encode())
            hash_int = int(hasher.hexdigest(), 16)
            chosen_member_index = hash_int % len(members)
            chosen_member = members[chosen_member_index]

            # get that user's details and preference
            chosen_user = session.get(User, chosen_member.user_id)
            if not chosen_user:
                continue

            preferred_hour = chosen_user.preferred_notification_hour_utc
            notification_hour = (
                preferred_hour
                if preferred_hour is not None
                else default_notification_hour
            )

            # calculate notification time (UTC - time zones are hard)
            notification_dt_utc = datetime.combine(
                today, time(hour=notification_hour, minute=0), tzinfo=timezone.utc
            )

            # if calculated time is in the past, schedule for now because we can't go back in time!
            if notification_dt_utc < now_utc:
                notification_dt_utc = now_utc

            bubble = session.get(Bubble, bubble_id)
            bubble_name = bubble.name if bubble else "Unknown Bubble"

            job_id = f"prompt_notify_{bubble_id}_{today.isoformat()}"
            try:
                scheduler.add_job(
                    sendNotification,
                    trigger="date",  # run once at specific date or  time
                    run_date=notification_dt_utc,
                    args=[  # Args for sendNotification
                        f"It's your turn to write the prompt for {bubble_name} today!",
                        [chosen_user.id],
                        "✏️ Your turn to write the prompt!",
                    ],
                    kwargs={},
                    id=job_id,  # unique ID for the job
                    replace_existing=True,  # overwrite if exists
                )
                print(
                    f"Scheduled prompt notification for user {chosen_user.id} at {notification_dt_utc} (Job ID: {job_id})"
                )
            except Exception as e:
                print(f"Error scheduling notification for user {chosen_user.id}: {e}")

        print("Finished scheduling prompt notifications.")
