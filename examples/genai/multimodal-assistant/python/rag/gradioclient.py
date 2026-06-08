import os
import time
import uuid
import json
import string
import random
import mimetypes
import requests
import sseclient
import backoff
import tarfile
import tempfile
import shutil
from pathlib import Path
from typing import Generator, Optional
from urllib.parse import urlparse, urlunparse
from rag.vectordb import RAG_DB_PATH

# ---------------- Utility Functions ----------------

def generate_session_hash(length=12) -> str:
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))

def generate_trigger_id() -> int:
    return int(time.time() * 1000) % 1000000

def guess_mime_type(path: str) -> str:
    return mimetypes.guess_type(path)[0] or "application/octet-stream"

class RagServerUnavailableError(Exception):
    pass

class FileUploadError(Exception):
    pass

class UploadProgressError(Exception):
    pass

class QueueJoinError(Exception):
    pass

class QueueProcessingError(Exception):
    pass

class DatabaseDownloadError(Exception):
    pass

# ---------------- Gradio Client ----------------

class GradioClient:
    def __init__(
        self,
        base_url: str,
        file_path: str,
        fn_index: int = 1,
        sync_url: Optional[str] = None,
        download_url: Optional[str] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.file_path = file_path
        self.fn_index = fn_index
        self.upload_id = str(uuid.uuid4())[:12]
        self.session_hash = generate_session_hash()
        self.trigger_id = generate_trigger_id()
        self.sync_url = sync_url or os.environ.get("RAG_SYNC_URL", "").strip() or None
        self.download_url = download_url or os.environ.get("RAG_DOWNLOAD_URL", "").strip() or None

    def check_server_available(self, timeout: int = 3) -> bool:
        """Check if the Gradio server is reachable before proceeding."""
        try:
            resp = requests.get(self.base_url + "/", timeout=timeout)
            return resp.status_code == 200
        except requests.RequestException:
            return False
        
    def build_file_payload(self, gradio_tmp_path: str) -> dict:
        return {
            "path": gradio_tmp_path,
            "url": f"{self.base_url}/gradio_api/file={gradio_tmp_path}",
            "orig_name": os.path.basename(self.file_path),
            "size": os.path.getsize(self.file_path),
            "mime_type": guess_mime_type(self.file_path),
            "meta": {"_type": "gradio.FileData"}
        }

    def upload_file(self) -> str:
        url = f"{self.base_url}/gradio_api/upload?upload_id={self.upload_id}"
        with open(self.file_path, "rb") as f:
            files = {"files": (os.path.basename(self.file_path), f)}
            resp = requests.post(url, files=files)
            resp.raise_for_status()
            return resp.json()[0]

    @backoff.on_exception(backoff.expo,
                          (requests.exceptions.RequestException, ValueError),
                          max_tries=5,
                          base=2)
    def wait_for_upload_progress(self):
        url = f"{self.base_url}/gradio_api/upload_progress?upload_id={self.upload_id}"
        print(f"🔁 Connecting to upload progress: {url}")
        resp = requests.get(url, stream=True, timeout=10)
        resp.encoding = "utf-8"
        if resp.status_code != 200:
            raise ValueError(f"Unexpected status code: {resp.status_code}")
        client = sseclient.SSEClient(resp)
        try:
            for event in client.events():
                data = json.loads(event.data)
                print("📡 Upload Progress:", data)
                if data.get("msg") == "done":
                    print("✅ Upload complete.")
                    break
        finally:
            client.close()

    def send_to_queue(self, gradio_tmp_path: str):
        url = f"{self.base_url}/gradio_api/queue/join"
        file_payload = self.build_file_payload(gradio_tmp_path)
        payload = {
            "data": [file_payload],
            "event_data": None,
            "fn_index": self.fn_index,
            "trigger_id": self.trigger_id,
            "session_hash": self.session_hash,
        }
        headers = {"Content-Type": "application/json"}
        resp = requests.post(url, json=payload, headers=headers)
        resp.raise_for_status()

    def trigger_sync(self) -> None:
        """Optional sync trigger for adapters that expose a dedicated sync endpoint."""
        if not self.sync_url:
            return
        resp = requests.post(self.sync_url, timeout=30)
        resp.raise_for_status()

    def _candidate_download_urls(self) -> list[str]:
        if self.download_url:
            return [self.download_url]

        parsed = urlparse(self.base_url)
        host = parsed.hostname or "127.0.0.1"
        scheme = parsed.scheme or "http"
        same_port = parsed.port

        candidates = []
        if same_port:
            candidates.append(f"{scheme}://{host}:{same_port}/download_db")
        candidates.append(f"{scheme}://{host}:8000/download_db")
        return candidates

    def download_and_extract_db(self) -> Path:
        """
        Download the database (tar.gz or raw .db) from backend API and return a local DB path.
        """
        tmp_dir = Path(tempfile.mkdtemp(prefix="ragdb_"))
        archive_path = tmp_dir / "milvus_db.tar.gz"
        raw_db_path = tmp_dir / "milvus.db"

        last_error = None
        for download_url in self._candidate_download_urls():
            try:
                with requests.get(download_url, stream=True, timeout=60) as resp:
                    resp.raise_for_status()
                    content_type = (resp.headers.get("content-type") or "").lower()
                    is_archive = "gzip" in content_type or "x-tar" in content_type or download_url.endswith(".tar.gz")

                    if is_archive:
                        with open(archive_path, "wb") as f:
                            for chunk in resp.iter_content(chunk_size=8192):
                                f.write(chunk)
                        with tarfile.open(archive_path, "r:gz") as tar:
                            tar.extractall(path=tmp_dir)
                        extracted_files = list(tmp_dir.rglob("*.db"))
                        if not extracted_files:
                            raise FileNotFoundError("Extracted DB file not found in archive.")
                        return extracted_files[0]

                    with open(raw_db_path, "wb") as f:
                        for chunk in resp.iter_content(chunk_size=8192):
                            f.write(chunk)
                    if raw_db_path.stat().st_size == 0:
                        raise ValueError("Downloaded DB file is empty.")
                    return raw_db_path
            except Exception as e:
                last_error = e
                continue

        raise DatabaseDownloadError(f"All DB download attempts failed: {last_error}")

def is_rag_fps_available(base_url: str = ""):
    client = GradioClient(base_url=base_url, file_path=None)
    return client.check_server_available()

def upload_and_process_file(
    file_path: str,
    base_url: str = "",
    sync_url: Optional[str] = None,
    download_url: Optional[str] = None,
) -> Generator[str, None, None]:
    client = GradioClient(
        base_url=base_url,
        file_path=file_path,
        sync_url=sync_url,
        download_url=download_url,
    )

    if not client.check_server_available():
        raise RagServerUnavailableError("RAG backend is unavailable. Please try again later.")

    # 1. Upload
    yield "📤 Uploading file..."
    try:
        gradio_tmp_path = client.upload_file()
    except Exception as e:
        raise FileUploadError(f"Upload failed: {str(e)}")

    # 2. Wait for upload to complete
    yield "⏳ Waiting for upload to complete..."
    try:
        client.wait_for_upload_progress()
    except Exception as e:
        raise UploadProgressError(f"Upload progress error: {str(e)}")

    # 3. Send to queue
    yield "📨 Sending to processing queue..."
    try:
        client.send_to_queue(gradio_tmp_path)
    except Exception as e:
        raise QueueJoinError(f"Queue join failed: {str(e)}")

    # 4. Listen for results
    yield "🎧 Listening for results..."
    try:
        url = f"{base_url}/gradio_api/queue/data?session_hash={client.session_hash}"
        resp = requests.get(url, stream=True, timeout=60)
        if resp.status_code != 200:
            raise QueueProcessingError(f"Queue stream error: {resp.status_code}")
        sse = sseclient.SSEClient(resp)

        for event in sse.events():
            if not event.data:
                continue
            data = json.loads(event.data)
            if data.get("msg") == "estimation":
                yield f"⏳ Queue position: {data.get('rank')}"
            elif data.get("msg") == "process_starts":
                yield "🚀 Processing started..."
            elif data.get("msg") == "process_completed":
                output = data.get("output", {})
                if data.get("success") and output.get("data"):
                    yield "✅ Processing Complete:"
                    for line in output["data"]:
                        yield str(line)
                else:
                    raise QueueProcessingError("Processing failed. Reason: " + (output.get("error") or "Unknown"))
                break
            elif data.get("msg") == "error":
                raise QueueProcessingError("Queue error: " + str(data))
    except Exception as e:
        raise QueueProcessingError(f"Listening error: {str(e)}")

    # 5. Download database
    yield "⏳ Retrieving database"

    try:
        if client.sync_url:
            yield "🔄 Triggering DB sync API..."
            client.trigger_sync()
        downloaded_file = client.download_and_extract_db()
        shutil.move(downloaded_file, RAG_DB_PATH)

        yield "⏳ Database synchronized"

    except Exception as e:
        raise DatabaseDownloadError(f"Database download error: {str(e)}")

# ---------------- Main for testing ----------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Upload a file to Gradio and monitor its processing.")
    parser.add_argument("base_url", help="Gradio base URL, e.g. https://abc123.gradio.live")
    parser.add_argument("file_path", help="Path to the file to upload")
    parser.add_argument("--sync-url", default=None, help="Optional FastAPI sync endpoint (POST)")
    parser.add_argument("--download-url", default=None, help="Optional DB download endpoint (GET)")
    args = parser.parse_args()

    from rag.gradioclient import upload_and_process_file  # if defined in another file

    for message in upload_and_process_file(
        file_path=args.file_path,
        base_url=args.base_url,
        sync_url=args.sync_url,
        download_url=args.download_url,
    ):
        print(message)

if __name__ == "__main__":
    main()
