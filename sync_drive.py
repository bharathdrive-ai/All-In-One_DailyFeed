import io
import json
import os
import re
import sys
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload


# ============================================================
# Download new PDFs from a Google Drive folder into documents/
#
# Env vars:
#   GDRIVE_FOLDER_ID            - ID from the folder's URL
#   GDRIVE_SERVICE_ACCOUNT_JSON - service account key (JSON text)
# ============================================================

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

# Keep dated PDFs (e.g. "... - 2026-09-29.pdf") for this many days.
# PDFs without a date in the name are always kept.
KEEP_DAYS = 7

DATE_PATTERN = re.compile(r"(\d{4}-\d{2}-\d{2})")


def get_cutoff():

    today = datetime.now(ZoneInfo("Asia/Kolkata")).date()

    # Today counts as one of the days
    return today - timedelta(days=KEEP_DAYS - 1)


def is_expired(name, cutoff):

    match = DATE_PATTERN.search(name)

    if not match:

        return False

    try:

        return date.fromisoformat(match.group(1)) < cutoff

    except ValueError:

        return False


def get_drive_service():

    info = json.loads(
        os.environ["GDRIVE_SERVICE_ACCOUNT_JSON"]
    )

    credentials = service_account.Credentials.from_service_account_info(
        info,
        scopes=SCOPES
    )

    return build("drive", "v3", credentials=credentials)


def list_pdfs(service, folder_id):

    files = []

    page_token = None

    while True:

        response = service.files().list(
            q=(
                f"'{folder_id}' in parents"
                " and mimeType='application/pdf'"
                " and trashed=false"
            ),
            fields="nextPageToken, files(id, name, size)",
            pageToken=page_token,
            supportsAllDrives=True,
            includeItemsFromAllDrives=True
        ).execute()

        files.extend(response.get("files", []))

        page_token = response.get("nextPageToken")

        if not page_token:

            return files


def download_file(service, file_id, path):

    request = service.files().get_media(
        fileId=file_id,
        supportsAllDrives=True
    )

    buffer = io.BytesIO()

    downloader = MediaIoBaseDownload(buffer, request)

    done = False

    while not done:

        _, done = downloader.next_chunk()

    with open(path, "wb") as f:

        f.write(buffer.getvalue())


def sync(folder="documents"):

    service = get_drive_service()

    os.makedirs(folder, exist_ok=True)

    files = list_pdfs(
        service,
        os.environ["GDRIVE_FOLDER_ID"]
    )

    print(f"PDFs in Drive folder: {len(files)}")

    cutoff = get_cutoff()

    print(f"Keeping dated PDFs from {cutoff} onwards")

    downloaded = 0

    for file in files:

        # Drive allows "/" in names; keep only the base name
        name = os.path.basename(file["name"])

        if is_expired(name, cutoff):

            continue

        path = os.path.join(folder, name)

        # Skip files we already have with the same size
        if (
            os.path.exists(path)
            and str(os.path.getsize(path)) == file.get("size")
        ):

            continue

        print(f"Downloading: {name}")

        download_file(service, file["id"], path)

        downloaded += 1

    print(f"New or updated PDFs: {downloaded}")

    removed = prune_old_pdfs(folder, cutoff)

    print(f"Old PDFs removed: {removed}")

    # Safety net: an empty listing may mean a sharing or API problem,
    # so never treat it as "everything was deleted"
    if files:

        missing = remove_missing_pdfs(
            folder,
            {os.path.basename(file["name"]) for file in files}
        )

        print(f"PDFs no longer in Drive removed: {missing}")

    else:

        print("Drive folder is empty; skipping mirror clean-up.")

    return downloaded


def remove_missing_pdfs(folder, drive_names):

    # Mirror the Drive folder: drop PDFs that were renamed or deleted there
    removed = 0

    for name in os.listdir(folder):

        if (
            name.lower().endswith(".pdf")
            and name not in drive_names
        ):

            print(f"Removing (not in Drive): {name}")

            os.remove(os.path.join(folder, name))

            removed += 1

    return removed


def prune_old_pdfs(folder, cutoff):

    removed = 0

    for name in os.listdir(folder):

        if (
            name.lower().endswith(".pdf")
            and is_expired(name, cutoff)
        ):

            print(f"Removing: {name}")

            os.remove(os.path.join(folder, name))

            removed += 1

    return removed


if __name__ == "__main__":

    missing = [
        var
        for var in ("GDRIVE_FOLDER_ID", "GDRIVE_SERVICE_ACCOUNT_JSON")
        if not os.environ.get(var)
    ]

    if missing:

        sys.exit(f"Missing env vars: {', '.join(missing)}")

    sync()
