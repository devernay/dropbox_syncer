#!/usr/bin/env python
"""
License:
dropbox_syncer
Copyright (C) 2025  Frédéric Devernay

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import argparse
import hashlib
import json
import os
import platform
import random
import sys
import time
import unicodedata
import uuid
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Callable, Dict, Optional, Set, Tuple, TypeVar
from typing import Optional, BinaryIO, Iterator
import io

import dropbox
import requests
from dropbox import DropboxOAuth2FlowNoRedirect
from dropbox.files import FileMetadata
from tqdm import tqdm

T = TypeVar("T")


def with_retry(
    max_attempts: int = 5,
    initial_delay: float = 2.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    exceptions=(
        ConnectionResetError,
        dropbox.exceptions.ApiError,
        requests.exceptions.ReadTimeout,
        requests.exceptions.ConnectionError,
        requests.exceptions.ChunkedEncodingError,  # Additional network error
    ),
) -> Callable:
    """
    Decorator that implements exponential backoff retry logic.

    Args:
        max_attempts: Maximum number of retry attempts
        initial_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        backoff_factor: Multiplicative factor for exponential backoff
        exceptions: Tuple of exceptions to catch and retry
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(self, *args, **kwargs) -> T:
            last_exception = None
            delay = initial_delay

            for attempt in range(max_attempts):
                try:
                    return func(self, *args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_attempts - 1:
                        # If this was the last attempt, re-raise the exception
                        raise

                    # Calculate next delay with jitter
                    jitter = random.uniform(0.8, 1.2)
                    next_delay = min(delay * backoff_factor * jitter, max_delay)

                    self._log_message(
                        f"Attempt {attempt + 1}/{max_attempts} failed: {str(e)}. "
                        f"Retrying in {delay:.2f} seconds..."
                    )

                    time.sleep(delay)
                    delay = next_delay

            # This should never be reached due to the raise in the last iteration
            assert last_exception is not None
            raise last_exception

        return wrapper

    return decorator


class DropboxSyncer:
    def __init__(
        self,
        auth_file: str,
        remote_folder: str,
        local_folder: str,
        app_key: Optional[str] = None,
        app_secret: Optional[str] = None,
        dry_run: bool = False,
        download: bool = True,
        upload: bool = False,
        overwrite: bool = False,
        chunked: bool = True,
    ):
        self.auth_file = auth_file
        self.remote_folder = remote_folder.rstrip("/")
        # Ensure the remote folder starts with '/' and doesn't end with '/', except for root
        self.normalized_remote_folder = "/" + self.remote_folder.strip("/")
        # With Dropbox API v2, the root is expressed as empty string "",
        # and non-root paths are expressed with a leading slash, e.g. "/test.txt".
        self.normalized_remote_folder_v2 = (
            self.normalized_remote_folder if self.normalized_remote_folder != "/" else ""
        )
        self.local_folder = Path(local_folder)
        self.provided_app_key = app_key
        self.provided_app_secret = app_secret
        self.dry_run = dry_run
        self.download = download
        self.upload = upload
        self.overwrite = overwrite
        self.chunked = chunked
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = f"dropbox_sync_log_{timestamp}.txt"
        self.dbx = self._authenticate()

    def _read_auth_file(self) -> Dict[str, str]:
        try:
            with open(self.auth_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    def _save_auth_file(self, data: Dict[str, str]):
        with open(self.auth_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    def _authenticate(self) -> dropbox.Dropbox:
        """Authenticate with Dropbox using OAuth2."""
        auth_data = self._read_auth_file()

        # Use provided credentials if available, otherwise fall back to auth file
        app_key = self.provided_app_key or auth_data.get("app_key")
        app_secret = self.provided_app_secret or auth_data.get("app_secret")
        refresh_token = auth_data.get("refresh_token")
        access_token = auth_data.get("access_token")

        if not app_key or not app_secret:
            app_key = input("Enter app key: ")
            app_secret = input("Enter app secret: ")
        auth_data["app_key"] = app_key
        auth_data["app_secret"] = app_secret

        if not (access_token and refresh_token):
            auth_flow = DropboxOAuth2FlowNoRedirect(
                app_key,
                app_secret,
                token_access_type="offline",  # This is important for getting refresh token
            )
            authorize_url = auth_flow.start()
            print("1. Go to:", authorize_url)
            print("2. Click 'Allow' (you might have to log in first)")
            print("3. Copy the authorization code")
            auth_code = input("Enter the authorization code: ").strip()

            try:
                oauth_result = auth_flow.finish(auth_code)
                access_token = oauth_result.access_token
                refresh_token = oauth_result.refresh_token

                auth_data.update(
                    {
                        "app_key": app_key,
                        "app_secret": app_secret,
                        "access_token": access_token,
                        "refresh_token": refresh_token,
                    }
                )
                self._save_auth_file(auth_data)
            except Exception as e:
                print(f"Error: {e}")
                sys.exit(1)

        return dropbox.Dropbox(
            oauth2_refresh_token=refresh_token, app_key=app_key, app_secret=app_secret
        )

    def _get_dropbox_content_hash(self, filepath: Path) -> str:
        """Calculate Dropbox content hash of a file."""
        BLOCK_SIZE = 4 * 1024 * 1024  # 4MB blocks

        block_hashes = []
        with open(filepath, "rb") as f:
            while True:
                block_data = f.read(BLOCK_SIZE)
                if not block_data:
                    break
                block_hash = hashlib.sha256(block_data).digest()
                block_hashes.append(block_hash)

        if not block_hashes:
            return hashlib.sha256(b"").hexdigest()

        combined_hash = hashlib.sha256(b"".join(block_hashes)).hexdigest()
        return combined_hash

    def _get_forbidden_chars(self) -> Set[str]:
        """Get set of characters that are forbidden in filenames for the current OS."""
        if platform.system() == "Windows":
            return {"<", ">", ":", '"', "\\", "|", "?", "*", "/"}
        elif platform.system() == "Darwin":  # macOS
            return {":"}
        else:  # Linux and others
            return {"/"}

    def _has_forbidden_chars(self, path_component: str) -> bool:
        """Check if a single path component (filename or folder name) contains forbidden characters."""
        forbidden_chars = self._get_forbidden_chars()
        return any(char in path_component for char in forbidden_chars)

    def _validate_path(self, path: Path, path_sep: str) -> bool:
        """Validate if the path contains any forbidden characters.

        Args:
            path: Path to validate
            path_sep: path separator

        Returns:
            bool: True if path is valid, False otherwise
        """
        components = str(path).split(path_sep)
        if any(self._has_forbidden_chars(component) for component in components):
            self._log_message(f"Skipped file due to forbidden characters: {path}")
            return False
        return True

    def _log_message(self, message: str) -> None:
        """Write a message to the log file with timestamp."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] {message}\n")

    def _is_case_sensitive_fs(self, path: Path) -> bool:
        """
        Check if the filesystem at the given path is case sensitive.
        """
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
        test_name = f"case_test_{uuid.uuid4().hex}"
        test_path = path / test_name

        try:
            test_path.touch()
            # If we can open the file with different case, filesystem is case-insensitive
            is_sensitive = not (path / test_name.upper()).exists()
            test_path.unlink()  # Clean up test file
            return is_sensitive
        except Exception:
            # If we can't create the test file, assume case-sensitive
            if test_path.exists():
                test_path.unlink()
            return True

    def _local_path_to_dropbox_path(self, local_path: Path) -> str:
        """
        Convert a local path to a Dropbox path.
        """
        # Convert the local path to relative path for comparison
        relative_path = local_path.relative_to(self.local_folder)

        # Normalize to NFC form for the dropbox path (Dropbox uses NFC, macOS uses NFD)
        return f"{self.normalized_remote_folder_v2}/{unicodedata.normalize('NFC', str(relative_path)).replace(os.sep, '/')}"

    def _normalize_case(
        self, local_path: Path, remote_files: Dict[str, FileMetadata]
    ) -> Tuple[Optional[Path], str]:
        """
        Returns the path with corrected case to match remote path if the filesystem is case-insensitive.
        Does not perform any actual file operations.
        Returns:
        A pair of two paths.
        First element is:
            - Path with corrected case if a case-insensitive match is found
            - Original path if exact match is found
            - None if no match is found
        Second element is the dropbox path.
        """
        # Check if we need case normalization
        if not hasattr(self, "_case_sensitive"):
            self._case_sensitive = self._is_case_sensitive_fs(self.local_folder)

        dropbox_path = self._local_path_to_dropbox_path(local_path)
        # print(f"Checking case for {local_path} ({dropbox_path})")

        # If filesystem is case-sensitive, return original path
        if self._case_sensitive:
            return local_path, dropbox_path

        # Try exact match first
        if dropbox_path in remote_files:
            return local_path, dropbox_path

        # Try case-insensitive match
        lower_dropbox_path = dropbox_path.lower()
        for remote_path in remote_files:
            if remote_path.lower() == lower_dropbox_path:
                # Found a match with different case
                remote_relative = remote_path[len(self.normalized_remote_folder) :].lstrip("/")
                corrected_path = self.local_folder / Path(remote_relative)

                # print(f"Renaming {local_path} to {corrected_path} ({dropbox_path} matches {remote_path})")
                return corrected_path, remote_path

        return None, dropbox_path

    def _validate_and_log_sync_operation(self) -> None:
        if self.download and self.upload:
            message = f"Starting two-way synchronization between Dropbox folder '{self.remote_folder}' and local folder '{self.local_folder}'"
        elif self.download:
            message = f"Starting synchronization from Dropbox folder '{self.remote_folder}' to local folder '{self.local_folder}'"
        elif self.upload:
            message = f"Starting synchronization from local folder '{self.local_folder}' to Dropbox folder '{self.remote_folder}'"
        else:
            message = "No action specified. Use --download or --upload."

        if self.dry_run:
            message = "[dry run] " + message
        print(message)
        self._log_message(message)

    CHUNK_SIZE = 8 * 1024 * 1024  # 8MB chunks for upload
    DOWNLOAD_CHUNK_SIZE = 8 * 1024 * 1024  # 8MB chunks for download

    def _upload_file_chunked(
        self, local_path: Path, dropbox_path: str, reason: Optional[str] = None
    ) -> None:
        """Upload a large file in chunks with retry logic."""
        file_size = local_path.stat().st_size

        message = f"Uploading {local_path} to {dropbox_path}"
        if reason is not None:
            message += f" ({reason})"
        if self.dry_run:
            self._log_message("[dry run] " + message)
            return
        self._log_message(message)

        try:
            with open(local_path, "rb") as f:
                if file_size <= self.CHUNK_SIZE:
                    # Small file - upload directly
                    self.dbx.files_upload(
                        f.read(), dropbox_path, mode=dropbox.files.WriteMode.overwrite
                    )
                else:
                    # Large file - upload in chunks with progress bar
                    with tqdm(total=file_size, desc="Uploading", unit="B", unit_scale=True) as pbar:
                        upload_session = self.dbx.files_upload_session_start(
                            f.read(self.CHUNK_SIZE)
                        )
                        cursor = dropbox.files.UploadSessionCursor(
                            session_id=upload_session.session_id, offset=f.tell()
                        )

                        # Upload chunks
                        while f.tell() < file_size - self.CHUNK_SIZE:
                            self.dbx.files_upload_session_append_v2(f.read(self.CHUNK_SIZE), cursor)
                            cursor.offset = f.tell()
                            pbar.update(self.CHUNK_SIZE)

                        # Upload final chunk and finish session
                        commit = dropbox.files.CommitInfo(
                            path=dropbox_path, mode=dropbox.files.WriteMode.overwrite
                        )
                        self.dbx.files_upload_session_finish(
                            f.read(self.CHUNK_SIZE), cursor, commit
                        )
                        pbar.update(file_size - pbar.n)  # Update to 100%

        except Exception as e:
            self._log_message(f"Error uploading {local_path}: {str(e)}")
            raise

    # Note: the following is not working, but it may not be useful anyway.
    def _download_file_chunked_broken(
        self, local_path: Path, dropbox_path: str, reason: Optional[str] = None
    ) -> None:
        """Download a large file in chunks with retry logic."""
        message = f"Downloading {dropbox_path} to {local_path}"
        if reason is not None:
            message += f" ({reason})"
        if self.dry_run:
            self._log_message("[dry run] " + message)
            return
        self._log_message(message)

        try:
            # Create parent directories if needed
            local_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                # Get file metadata first
                metadata = self.dbx.files_get_metadata(dropbox_path)
                if not isinstance(metadata, dropbox.files.FileMetadata):
                    raise ValueError(f"Path is not a file: {dropbox_path}")

                file_size = metadata.size

                with open(local_path, "wb") as f:
                    if file_size <= self.DOWNLOAD_CHUNK_SIZE:
                        # Small file - download directly
                        metadata, response = self.dbx.files_download(dropbox_path)
                        f.write(response.content)
                    else:
                        with tqdm(
                            total=file_size, desc="Downloading", unit="B", unit_scale=True
                        ) as pbar:
                            # Large file - download in chunks with progress bar
                            for chunk_start in range(0, file_size, self.DOWNLOAD_CHUNK_SIZE):
                                chunk_end = min(
                                    chunk_start + self.DOWNLOAD_CHUNK_SIZE - 1, file_size - 1
                                )
                                ## range_start and range_end not available in the Python API v2
                                # metadata, response = self.dbx.files_download(
                                #     dropbox_path,
                                #     range_start=chunk_start,
                                #     range_end=chunk_end
                                # )

                                # Get download link first
                                metadata, response = self.dbx.files_download(dropbox_path)
                                download_url = response.url

                                # Download chunk with range header
                                headers = {"Range": f"bytes={chunk_start}-{chunk_end}"}
                                response = self.dbx.session.get(download_url, headers=headers)
                                response.raise_for_status()

                                chunk_data = response.content
                                f.write(chunk_data)
                                pbar.update(len(chunk_data))

            except dropbox.exceptions.ApiError as e:
                if (
                    isinstance(e.error, dropbox.files.DownloadError)
                    and e.error.is_unsupported_file()
                ):
                    try:
                        # Try to export the file instead
                        self.dbx.files_export_to_file(str(local_path), dropbox_path)
                        return
                    except dropbox.exceptions.ApiError as export_error:
                        self._log_message(f"Error exporting {dropbox_path}: {str(export_error)}")
                        raise
                self._log_message(f"Error downloading {dropbox_path}: {str(e)}")
                raise

        except Exception as e:
            # Clean up partial downloads on error
            if local_path.exists():
                local_path.unlink()
            raise

    @with_retry(max_attempts=3, initial_delay=1.0, max_delay=30.0)
    def _upload_file(
        self, local_path: Path, dropbox_path: str, reason: Optional[str] = None
    ) -> None:
        """Upload a file to Dropbox with retry logic."""

        if self.chunked:
            return self._upload_file_chunked(local_path, dropbox_path, reason)

        message = f"Uploading {local_path} to {dropbox_path}"
        if reason is not None:
            message += f" ({reason})"
        if self.dry_run:
            self._log_message("[dry run] " + message)
            return
        self._log_message(message)
        try:
            with open(local_path, "rb") as f:
                self.dbx.files_upload(
                    f.read(), dropbox_path, mode=dropbox.files.WriteMode.overwrite
                )
        except Exception as e:
            self._log_message(f"Error uploading {local_path}: {str(e)}")
            raise

    @with_retry(max_attempts=3, initial_delay=1.0, max_delay=30.0)
    def _download_file(
        self, local_path: Path, dropbox_path: str, reason: Optional[str] = None
    ) -> None:
        """Download a file from Dropbox with retry logic."""

        # if self.chunked:
        #    return self._download_file_chunked(local_path, dropbox_path, reason)

        message = f"Downloading {dropbox_path} to {local_path}"
        if reason is not None:
            message += f" ({reason})"
        if self.dry_run:
            self._log_message("[dry run] " + message)
            return
        self._log_message(message)
        try:
            self.dbx.files_download_to_file(str(local_path), dropbox_path)
        except dropbox.exceptions.ApiError as e:
            if isinstance(e.error, dropbox.files.DownloadError) and e.error.is_unsupported_file():
                try:
                    # Try to export the file instead
                    self.dbx.files_export_to_file(str(local_path), dropbox_path)
                    return
                except dropbox.exceptions.ApiError:
                    self._log_message(f"Error exporting {dropbox_path}: {str(e)}")
                    raise
            self._log_message(f"Error downloading {dropbox_path}: {str(e)}")
            raise

    def sync(self) -> None:
        """Perform the two-way synchronization between Dropbox and a local folder."""
        self._validate_and_log_sync_operation()

        self.local_folder.mkdir(parents=True, exist_ok=True)

        # Create a dictionary of remote files with their metadata
        remote_files = {}
        try:
            cursor = None
            while True:
                if cursor is None:
                    try:
                        result = self.dbx.files_list_folder(
                            self.normalized_remote_folder_v2, recursive=True
                        )
                    except dropbox.exceptions.ApiError as e:
                        if isinstance(e.error, dropbox.files.ListFolderError):
                            try:
                                self.dbx.files_get_metadata(self.normalized_remote_folder_v2)
                            except dropbox.exceptions.ApiError as metadata_error:
                                print(
                                    f"Error: Cannot access folder '{self.normalized_remote_folder}'."
                                )
                                self._log_message(
                                    f"Error accessing remote folder: {str(metadata_error)}"
                                )
                                return
                        raise
                else:
                    result = self.dbx.files_list_folder_continue(cursor)

                for entry in result.entries:
                    if isinstance(entry, FileMetadata):
                        # Find the case-insensitive position of the remote folder in the path
                        entry_path = entry.path_display
                        if entry_path.lower().startswith(self.normalized_remote_folder.lower()):
                            # Replace the remote folder part while preserving the rest of the path
                            normalized_path = (
                                self.normalized_remote_folder
                                + entry_path[len(self.normalized_remote_folder) :]
                            )
                            remote_files[normalized_path] = entry

                if not result.has_more:
                    break
                cursor = result.cursor

        except dropbox.exceptions.AuthError:
            print("Authentication error. Please check your authentication tokens.")
            self._log_message("Authentication error occurred during synchronization")
            return
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            self._log_message(f"Unexpected error: {str(e)}")
            return

        # Count total operations before starting the progress bar
        total_operations = 0
        local_files_new = 0
        local_files_illegal = 0
        local_files_exist_in_dropbox = 0

        # Count local files
        for root, _, files in os.walk(self.local_folder):
            for file in files:
                local_path = Path(root) / file
                # Normalize the case of the path
                normalized_path, dropbox_path = self._normalize_case(local_path, remote_files)
                if normalized_path is None:
                    # This is a new local file
                    normalized_path = local_path

                relative_path = local_path.relative_to(self.local_folder)
                # Skip files with forbidden characters
                if not self._validate_path(relative_path, os.sep):
                    local_files_illegal += 1
                    continue

                if dropbox_path not in remote_files:
                    total_operations += 1
                    local_files_new += 1
                    # print(f"New local file: {local_path} {dropbox_path}")
                else:
                    local_files_exist_in_dropbox += 1

        remote_files_new = 0
        remote_files_illegal = 0
        # Count remaining remote files (files that exist only in Dropbox)
        for dropbox_path in remote_files:
            relative_path = dropbox_path[len(self.normalized_remote_folder) :].lstrip("/")
            if not self._validate_path(relative_path, "/"):
                remote_files_illegal += 1
                continue
            total_operations += 1
            remote_files_new += 1
        remote_files_new -= local_files_exist_in_dropbox

        self._log_message(f"Total operations to process: {total_operations}")
        self._log_message("Including:")
        self._log_message(f"- {local_files_new} new local files,")
        self._log_message(f"- {local_files_exist_in_dropbox} files both in local and Dropbox")
        self._log_message(f"- {remote_files_new} new Dropbox files")
        self._log_message("Excluding:")
        self._log_message(f"- {local_files_illegal} local files with illegal filenames,")
        self._log_message(f"- {remote_files_illegal} Dropbox files with illegal filenames.")

        # Process files with the progress bar
        with tqdm(total=total_operations, desc="Syncing files") as pbar:
            for root, _, files in os.walk(self.local_folder):
                for file in files:
                    local_path = Path(root) / file
                    # Normalize the case of the path
                    normalized_path, dropbox_path = self._normalize_case(local_path, remote_files)
                    if normalized_path is None:
                        # This is a new local file
                        normalized_path = local_path

                    relative_path = normalized_path.relative_to(self.local_folder)
                    assert dropbox_path is not None

                    # Check for forbidden characters
                    if not self._validate_path(relative_path, os.sep):
                        self._log_message(
                            f"Skipped local file due to forbidden characters: {local_path}"
                        )
                        continue

                    if dropbox_path in remote_files:
                        # File exists in both places - compare content and timestamps
                        local_hash = self._get_dropbox_content_hash(local_path)
                        remote_entry = remote_files[dropbox_path]

                        if local_hash != remote_entry.content_hash:
                            if not self.overwrite:
                                self._log_message(
                                    "Skipped file because it exists both locally and "
                                    f"remotely, although hashes differ: {local_path}"
                                )
                            else:
                                # Hashes differ, compare modification time
                                local_mtime = local_path.stat().st_mtime
                                remote_mtime = remote_entry.server_modified.timestamp()
                                if remote_mtime > local_mtime:
                                    if self.download:
                                        # Remote is newer - download
                                        self._download_file(
                                            local_path, dropbox_path, "remote file is newer"
                                        )
                                elif self.upload:
                                    # Local is newer - upload
                                    self._upload_file(
                                        local_path, dropbox_path, "local file is newer"
                                    )

                        # Remove from remote_files dict to track processed files
                        del remote_files[dropbox_path]
                    elif self.upload:
                        # File exists only locally - upload it
                        self._upload_file(local_path, dropbox_path, "file only exists locally")

                    pbar.update(1)

            # Download remaining files that exist only in Dropbox
            for dropbox_path, entry in remote_files.items():
                relative_path = entry.path_display[len(self.normalized_remote_folder) :].lstrip("/")
                local_path = self.local_folder / relative_path

                # Check for forbidden characters
                if not self._validate_path(relative_path, "/"):
                    self._log_message(
                        f"Skipped remote file due to forbidden characters: {entry.path_display}"
                    )
                    continue

                if self.download:
                    # Create parent directories if needed
                    local_path.parent.mkdir(parents=True, exist_ok=True)

                    self._download_file(local_path, dropbox_path, "file does not exist locally")
                pbar.update(1)

        message = f"Ended synchronization between {self.remote_folder} and {self.local_folder}"
        if self.dry_run:
            message = "[dry run] " + message
        print(message)
        self._log_message(message)


def main():
    parser = argparse.ArgumentParser(description="Dropbox Two-Way Sync Tool")
    parser.add_argument("--remote-folder", default="/", help="Dropbox folder path")
    parser.add_argument("--local-folder", default="D:/Dropbox", help="Local folder path")
    parser.add_argument("--auth-file", default="dropbox_auth.json", help="Authentication file path")
    parser.add_argument("--app-key", help="Dropbox app key (optional)")
    parser.add_argument("--app-secret", help="Dropbox app secret (optional)")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not do any actual action, only write the log file of what would be done.",
    )
    parser.add_argument("-D", "--download", action="store_true", help="Download from Dropbox.")
    parser.add_argument("-U", "--upload", action="store_true", help="Upload to Dropbox.")
    parser.add_argument(
        "-O",
        "--overwrite",
        action="store_true",
        help=(
            "If file exists at both places but differ, overwrite either version "
            "based on their modification time."
        ),
    )

    args = parser.parse_args()

    if not (args.download or args.upload):
        parser.error("No action requested, add -D and/or -U options")

    syncer = DropboxSyncer(
        auth_file=args.auth_file,
        remote_folder=args.remote_folder,
        local_folder=args.local_folder,
        app_key=args.app_key,
        app_secret=args.app_secret,
        dry_run=args.dry_run,
        download=args.download,
        upload=args.upload,
        overwrite=args.overwrite,
    )
    syncer.sync()


if __name__ == "__main__":
    main()
