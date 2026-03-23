#!/usr/bin/env python3
"""One-time setup for Google Calendar API.

Interactive script that guides through the entire setup:
  1. Creates Google Cloud project (opens browser)
  2. Enables Calendar API (opens browser)
  3. Creates OAuth credentials (opens browser)
  4. Runs OAuth consent flow
  5. Tests the connection

Usage:
  python setup_calendar.py
"""

import os
import sys
import webbrowser
from pathlib import Path

# Allow OAuth over http://localhost (required for desktop app flow)
os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

from calendar_client import SCOPES

CREDENTIALS_FILE = Path(__file__).parent / "client_secret.json"
TOKEN_FILE = Path(__file__).parent / "token.json"

PROJECT_NAME = "voice-assistant"


def wait_for_enter(prompt="Press Enter to continue..."):
    input(f"\n{prompt}")


def step_create_project():
    """Guide user through Google Cloud project creation."""
    print("=" * 60)
    print("STEP 1: Create a Google Cloud project")
    print("=" * 60)
    print()
    print("A browser window will open to create a new project.")
    print(f'Name it "{PROJECT_NAME}" (or anything you like).')
    wait_for_enter("Press Enter to open Google Cloud Console...")

    webbrowser.open("https://console.cloud.google.com/projectcreate")
    wait_for_enter("Done? Press Enter when the project is created...")


def step_enable_api():
    """Guide user through enabling the Calendar API."""
    print()
    print("=" * 60)
    print("STEP 2: Enable Google Calendar API")
    print("=" * 60)
    print()
    print('A browser window will open. Click "ENABLE".')
    print("(Make sure the correct project is selected at the top.)")
    wait_for_enter("Press Enter to open the Calendar API page...")

    webbrowser.open("https://console.cloud.google.com/apis/library/calendar-json.googleapis.com")
    wait_for_enter("Done? Press Enter when the API is enabled...")


def step_consent_screen(project_id: str | None = None):
    """Guide user through OAuth consent screen setup."""
    print()
    print("=" * 60)
    print("STEP 3: Configure OAuth consent screen")
    print("=" * 60)
    print()
    print("Two pages will open:")
    print()
    print("  Page 1 — Branding (consent screen):")
    print("    - User type: External")
    print('    - App name: "Voice Assistant" (or anything)')
    print("    - User support email: your email")
    print("    - Developer contact: your email")
    print("    - Leave everything else blank, click Save")
    print()
    print("  Page 2 — Audience (test users):")
    print('    - Click "Add users"')
    print("    - Enter your Gmail address")
    print("    - Save")

    project_param = f"?project={project_id}" if project_id else ""

    wait_for_enter("Press Enter to open the branding page...")
    webbrowser.open(f"https://console.cloud.google.com/auth/branding{project_param}")
    wait_for_enter("Branding saved? Press Enter to open the audience page...")
    webbrowser.open(f"https://console.cloud.google.com/auth/audience{project_param}")
    wait_for_enter("Done? Press Enter when your email is added as test user...")


def step_create_credentials():
    """Guide user through OAuth credential creation."""
    print()
    print("=" * 60)
    print("STEP 4: Create OAuth credentials")
    print("=" * 60)
    print()
    print("A browser window will open to create credentials.")
    print()
    print('  1. Click "CREATE CREDENTIALS" > "OAuth client ID"')
    print('  2. Application type: "Desktop app"')
    print(f'  3. Name: "{PROJECT_NAME}"')
    print('  4. Click "CREATE"')
    print('  5. Click "DOWNLOAD JSON"')
    print(f"  6. Save the file as: {CREDENTIALS_FILE}")
    wait_for_enter("Press Enter to open the Credentials page...")

    webbrowser.open("https://console.cloud.google.com/apis/credentials")
    print()
    print(f"Waiting for {CREDENTIALS_FILE.name} ...")

    while not CREDENTIALS_FILE.exists():
        wait_for_enter(f"{CREDENTIALS_FILE.name} not found. Download it and press Enter...")

    print(f"Found {CREDENTIALS_FILE.name}!")


def step_oauth_flow():
    """Run the OAuth consent flow."""
    from google_auth_oauthlib.flow import InstalledAppFlow

    print()
    print("=" * 60)
    print("STEP 5: Authorize access to your Google Calendar")
    print("=" * 60)
    print()
    print("A browser window will open for Google sign-in.")
    print("Sign in and grant calendar access.")
    print()

    flow = InstalledAppFlow.from_client_secrets_file(str(CREDENTIALS_FILE), SCOPES)
    creds = flow.run_local_server(port=0)

    TOKEN_FILE.write_text(creds.to_json())
    print(f"Token saved to {TOKEN_FILE}")
    return creds


def step_test_connection(creds):
    """Test the connection by listing upcoming events."""
    from datetime import datetime, timezone

    from googleapiclient.discovery import build

    print()
    print("=" * 60)
    print("STEP 6: Testing connection")
    print("=" * 60)
    print()

    service = build("calendar", "v3", credentials=creds)

    now = datetime.now(timezone.utc).isoformat()
    result = (
        service.events()
        .list(
            calendarId="primary",
            timeMin=now,
            maxResults=5,
            singleEvents=True,
            orderBy="startTime",
            fields="items(summary,start)",
        )
        .execute()
    )

    events = result.get("items", [])
    if events:
        print(f"Found {len(events)} upcoming events:")
        for event in events:
            start = event["start"].get("dateTime", event["start"].get("date"))
            print(f"  - {event.get('summary', '(no title)')} @ {start}")
    else:
        print("No upcoming events found (calendar may be empty).")

    print()
    print("Setup complete! The voice assistant can now access your calendar.")
    print("Restart the server: ./ctl.sh restart")


def main():
    try:
        from google_auth_oauthlib.flow import InstalledAppFlow  # noqa: F401
        from googleapiclient.discovery import build  # noqa: F401
    except ImportError:
        print("Missing dependencies. Install with:")
        print("  pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib")
        sys.exit(1)

    print()
    print("Google Calendar Setup for Voice Assistant")
    print("==========================================")
    print()

    if TOKEN_FILE.exists():
        print(f"Token already exists ({TOKEN_FILE}).")
        resp = input("Re-run setup from scratch? [y/N] ").strip().lower()
        if resp != "y":
            print("Aborted.")
            return

    if CREDENTIALS_FILE.exists():
        print(f"Found existing {CREDENTIALS_FILE.name}, skipping steps 1-4.")
    else:
        step_create_project()
        step_enable_api()
        step_consent_screen()
        step_create_credentials()

    creds = step_oauth_flow()
    step_test_connection(creds)


if __name__ == "__main__":
    main()
