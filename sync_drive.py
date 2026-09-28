import io
import json
import os
import sys

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

    downloaded = 0

    for file in files:

        # Drive allows "/" in names; keep only the base name
        name = os.path.basename(file["name"])

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

    return downloaded


if __name__ == "__main__":

    missing = [
        var
        for var in ("GDRIVE_FOLDER_ID", "GDRIVE_SERVICE_ACCOUNT_JSON")
        if not os.environ.get(var)
    ]

    if missing:

        sys.exit(f"Missing env vars: {', '.join(missing)}")

    sync()
