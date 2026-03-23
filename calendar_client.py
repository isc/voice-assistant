"""Google Calendar client for the voice assistant.

Provides async methods to query and create calendar events using the
Google Calendar API. The underlying google-api-python-client is synchronous,
so all API calls run in a thread pool via asyncio.to_thread().
"""

import asyncio
import json
import locale
import logging
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

try:
    locale.setlocale(locale.LC_TIME, "fr_FR.UTF-8")
except locale.Error:
    pass  # Fall back to system locale

logger = logging.getLogger(__name__)

SCOPES = ["https://www.googleapis.com/auth/calendar.events"]
TIMEZONE = "Europe/Paris"
TZ = ZoneInfo(TIMEZONE)

LIST_FIELDS = "items(id,summary,start,end,status)"


class CalendarClient:
    """Async wrapper around the Google Calendar API."""

    def __init__(self, credentials_path: str, token_path: str):
        from pathlib import Path

        self._credentials_path = Path(credentials_path)
        self._token_path = Path(token_path)
        self._service = None

    async def connect(self) -> bool:
        """Load OAuth credentials and build the Calendar API service."""
        try:
            creds = await asyncio.to_thread(self._load_credentials)
            if creds is None:
                return False

            from googleapiclient.discovery import build

            self._service = await asyncio.to_thread(build, "calendar", "v3", credentials=creds)

            now = datetime.now(timezone.utc).isoformat()
            await asyncio.to_thread(
                self._service.events().list(calendarId="primary", timeMin=now, maxResults=1).execute
            )

            return True
        except Exception as e:
            logger.error(f"Google Calendar connection failed: {e}")
            return False

    def _load_credentials(self):
        """Load and refresh OAuth credentials (runs in thread)."""
        from google.auth.exceptions import RefreshError
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials

        if not self._token_path.exists():
            logger.info("No calendar token found. Run setup_calendar.py to authorize.")
            return None

        creds = Credentials.from_authorized_user_file(str(self._token_path), SCOPES)

        if creds.valid:
            return creds

        if creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
                # Save refreshed token
                self._token_path.write_text(creds.to_json())
                return creds
            except RefreshError:
                logger.error("Calendar token expired or revoked. Run setup_calendar.py to re-authorize.")
                return None

        logger.error("Calendar token invalid. Run setup_calendar.py to re-authorize.")
        return None

    async def close(self):
        """Clean up the API service."""
        if self._service:
            self._service.close()
            self._service = None

    async def query_events(
        self,
        start_date: str,
        end_date: str | None = None,
        search: str | None = None,
    ) -> str:
        """Query calendar events for a date range.

        Args:
            start_date: Start date in YYYY-MM-DD format.
            end_date: End date in YYYY-MM-DD format. Defaults to start_date.
            search: Optional text to search in event titles.

        Returns:
            JSON string with events list for the LLM.
        """
        if not self._service:
            return json.dumps({"error": "Agenda non disponible"})

        try:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=TZ)
        except ValueError:
            return json.dumps({"error": f"Date invalide: {start_date} (format attendu: AAAA-MM-JJ)"})

        if end_date:
            try:
                end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=TZ)
                # Include the full end day
                end_dt = end_dt + timedelta(days=1)
            except ValueError:
                return json.dumps({"error": f"Date invalide: {end_date} (format attendu: AAAA-MM-JJ)"})
        else:
            end_dt = start_dt + timedelta(days=1)

        time_min = start_dt.isoformat()
        time_max = end_dt.isoformat()

        try:
            result = await asyncio.to_thread(self._list_events, time_min, time_max, search)
        except Exception as e:
            logger.error(f"Calendar query failed: {e}")
            return json.dumps({"error": "Erreur lors de la consultation de l'agenda"})

        events = []
        for item in result.get("items", []):
            if item.get("status") == "cancelled":
                continue
            events.append(_format_event(item))

        # Build period description in French
        if end_date and end_date != start_date:
            period = f"du {_format_date_fr(start_dt)} au {_format_date_fr(end_dt - timedelta(days=1))}"
        else:
            period = _format_date_fr(start_dt)

        return json.dumps(
            {"events": events, "period": period, "count": len(events)},
            ensure_ascii=False,
        )

    def _list_events(self, time_min: str, time_max: str, search: str | None) -> dict:
        """Execute the events.list API call (runs in thread)."""
        kwargs = {
            "calendarId": "primary",
            "timeMin": time_min,
            "timeMax": time_max,
            "maxResults": 250,
            "singleEvents": True,
            "orderBy": "startTime",
            "fields": LIST_FIELDS,
            "timeZone": TIMEZONE,
        }
        if search:
            kwargs["q"] = search

        return self._service.events().list(**kwargs).execute()

    async def create_event(
        self,
        title: str,
        start_datetime: str,
        duration_minutes: int = 60,
    ) -> str:
        """Create a calendar event.

        Args:
            title: Event title.
            start_datetime: Start in "YYYY-MM-DD HH:MM" format.
            duration_minutes: Duration in minutes (default 60).

        Returns:
            JSON string with confirmation for the LLM.
        """
        if not self._service:
            return json.dumps({"error": "Agenda non disponible"})

        try:
            start_dt = datetime.strptime(start_datetime, "%Y-%m-%d %H:%M").replace(tzinfo=TZ)
        except ValueError:
            return json.dumps({"error": f"Date/heure invalide: {start_datetime} (format attendu: AAAA-MM-JJ HH:MM)"})

        end_dt = start_dt + timedelta(minutes=duration_minutes)

        event_body = {
            "summary": title,
            "start": {"dateTime": start_dt.isoformat(), "timeZone": TIMEZONE},
            "end": {"dateTime": end_dt.isoformat(), "timeZone": TIMEZONE},
        }

        try:
            created = await asyncio.to_thread(self._insert_event, event_body)
        except Exception as e:
            logger.error(f"Calendar event creation failed: {e}")
            return json.dumps({"error": "Erreur lors de la création de l'événement"})

        # Format duration for display
        if duration_minutes >= 60:
            hours = duration_minutes // 60
            mins = duration_minutes % 60
            dur_str = f"{hours}h" + (f"{mins:02d}" if mins else "")
        else:
            dur_str = f"{duration_minutes}min"

        return json.dumps(
            {
                "status": "created",
                "title": created.get("summary", title),
                "start": _format_datetime_fr(start_dt),
                "duration": dur_str,
            },
            ensure_ascii=False,
        )

    def _insert_event(self, event_body: dict) -> dict:
        """Execute the events.insert API call (runs in thread)."""
        return self._service.events().insert(calendarId="primary", body=event_body).execute()


def _format_event(item: dict) -> dict:
    """Format a Google Calendar event item for the LLM."""
    start = item.get("start", {})
    end = item.get("end", {})

    if "dateTime" in start:
        start_dt = datetime.fromisoformat(start["dateTime"])
        start_str = start_dt.strftime("%H:%M")
        date_str = start_dt.strftime("%Y-%m-%d")
    else:
        start_str = "toute la journée"
        date_str = start.get("date", "")

    if "dateTime" in end:
        end_dt = datetime.fromisoformat(end["dateTime"])
        end_str = end_dt.strftime("%H:%M")
    else:
        end_str = ""

    result = {
        "title": item.get("summary", "(sans titre)"),
        "date": date_str,
        "start": start_str,
    }
    if end_str:
        result["end"] = end_str

    return result


def _format_date_fr(dt: datetime) -> str:
    """Format a date in French using locale (e.g., 'mercredi 25 mars 2026')."""
    return dt.strftime("%A %d %B %Y")


def _format_datetime_fr(dt: datetime) -> str:
    """Format a datetime in French (e.g., 'mercredi 25 mars 2026 à 14:00')."""
    return dt.strftime("%A %d %B %Y à %H:%M")
