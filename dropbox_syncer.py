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

Description:
Dropbox syncer using the Python API.
Synchronize a local directory with a Dropbox folder
using the Dropbox Python SDK.
Create a Scoped App from https://www.dropbox.com/developers/, using permissions:
files.content.write and files.content.read.
If you just want to download from Dropbox (option -D), only files.content.read is necessary.
If you need access to only one folder, you can create a scoped app with an App Folder.
"""
import argparse
import os
import sys
import json
import hashlib
import platform
from typing import Dict, List, Optional, Set, Tuple
from pathlib import Path
from datetime import datetime
import dropbox
from dropbox import DropboxOAuth2FlowNoRedirect
from dropbox.files import FileMetadata, FolderMetadata
from tqdm import tqdm

class DropboxSyncer:
    def __init__(
        self,
        auth_file: str,
        remote_folder: str,
        local_folder: str,
        app_key: Optional[str] = None,
        app_secret: Optional[str] = None
    ):
        self.auth_file = auth_file
        self.remote_folder = remote_folder.rstrip('/')
        self.local_folder = Path(local_folder)
        self.provided_app_key = app_key
        self.provided_app_secret = app_secret
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = f"dropbox_sync_log_{timestamp}.txt"
        self.dbx = self._authenticate()

    def _get_forbidden_chars(self) -> Set[str]:
        """Get set of characters that are forbidden in filenames for the current OS."""
        if platform.system() == 'Windows':
            return {'<', '>', ':', '"', '\\', '|', '?', '*', '/'}
        elif platform.system() == 'Darwin':  # macOS
            return {':'} 
        else:  # Linux and others
            return {'/'} 

    def _read_auth_file(self) -> Dict[str, str]:
        try:
            with open(self.auth_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    def _save_auth_file(self, data: Dict[str, str]):
        with open(self.auth_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)

    def _authenticate(self) -> dropbox.Dropbox:
        """Authenticate with Dropbox using OAuth2."""
        auth_data = self._read_auth_file()
        
        # Use provided credentials if available, otherwise fall back to auth file
        app_key = self.provided_app_key or auth_data.get('app_key')
        app_secret = self.provided_app_secret or auth_data.get('app_secret')
        refresh_token = auth_data.get('refresh_token')
        access_token = auth_data.get('access_token')

        if not app_key or not app_secret:
            app_key = input("Enter app key: ")
            app_secret = input("Enter app secret: ")
        auth_data['app_key'] = app_key
        auth_data['app_secret'] = app_secret

        if not (access_token and refresh_token):
            auth_flow = DropboxOAuth2FlowNoRedirect(
                app_key, 
                app_secret,
                token_access_type='offline'  # This is important for getting refresh token
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
                
                auth_data.update({
                    'app_key': app_key,
                    'app_secret': app_secret,
                    'access_token': access_token,
                    'refresh_token': refresh_token
                })
                self._save_auth_file(auth_data)
            except Exception as e:
                print(f"Error: {e}")
                sys.exit(1)

        return dropbox.Dropbox(
            oauth2_refresh_token=refresh_token,
            app_key=app_key,
            app_secret=app_secret
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
        path_components = str(path).split(path_sep)
        if any(self._has_forbidden_chars(component) for component in components):
            self._log_message(f"Skipped file due to forbidden characters: {path}")
            return False
        return True

    def _log_message(self, message: str) -> None:
        """Write a message to the log file with timestamp."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] {message}\n")

    def _is_case_sensitive_fs(self, path: Path) -> bool:
        """
        Check if the filesystem at the given path is case sensitive.
        """
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            
        test_file = path / "CaSeTeSt.txt"
        try:
            test_file.touch()
            # If we can open the file with different case, filesystem is case-insensitive
            is_sensitive = not (path / "casetest.txt").exists()
            test_file.unlink()  # Clean up test file
            return is_sensitive
        except Exception:
            # If we can't create the test file, assume case-sensitive
            return True

    def _normalize_case(self, local_path: Path, remote_files: Dict[str, FileMetadata]) -> Optional[Path]:
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
        if not hasattr(self, '_case_sensitive'):
            self._case_sensitive = self._is_case_sensitive_fs(self.local_folder)
        
        # Convert the local path to relative path for comparison
        relative_path = local_path.relative_to(self.local_folder)
        normalized_remote_folder = '/' + self.remote_folder.strip('/')

        dropbox_path = f"{normalized_remote_folder}/{str(relative_path).replace(os.sep, '/')}"

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
                remote_relative = remote_path[len(normalized_remote_folder):].lstrip('/')
                corrected_path = self.local_folder / Path(remote_relative)

                #print(f"Renaming {local_path} to {corrected_path} ({dropbox_path} matches {remote_path})")
                return corrected_path, remote_path
        
        return None, dropbox_path

    def _validate_and_log_sync_operation(self, download: bool, upload: bool) -> None:
        if download and upload:
            message = f"Starting two-way synchronization between Dropbox folder '{self.remote_folder}' and local folder '{self.local_folder}'"
        elif download:
            message = f"Starting synchronization from Dropbox folder '{self.remote_folder}' to local folder '{self.local_folder}'"
        elif upload:
            message = f"Starting synchronization from local folder '{self.local_folder}' to Dropbox folder '{self.remote_folder}'"
        else:
            message = "No action specified. Use --download or --upload."
        
        print(message)
        self._log_message(message)



    def sync(self, dry_run: bool=False, download: bool=True, upload: bool=False) -> None:
        """Perform the two-way synchronization between Dropbox and a local folder."""
        self._validate_and_log_sync_operation(download, upload)

        self.local_folder.mkdir(parents=True, exist_ok=True)

        # Ensure the remote folder starts with '/' and doesn't end with '/', except for root
        normalized_remote_folder = '/' + self.remote_folder.strip('/')

        # Create a dictionary of remote files with their metadata
        remote_files = {}
        try:
            cursor = None
            while True:
                if cursor is None:
                    # With Dropbox API v2, the root is expressed as empty string "",
                    # and non-root paths are expressed with a leading slash, e.g. "/test.txt".
                    normalized_remote_folder_v2 = normalized_remote_folder if normalized_remote_folder != "/" else ""
                    try:
                        result = self.dbx.files_list_folder(normalized_remote_folder_v2, recursive=True)
                    except dropbox.exceptions.ApiError as e:
                        if isinstance(e.error, dropbox.files.ListFolderError):
                            try:
                                self.dbx.files_get_metadata(normalized_remote_folder_v2)
                            except dropbox.exceptions.ApiError as metadata_error:
                                print(f"Error: Cannot access folder '{normalized_remote_folder}'.")
                                self._log_message(f"Error accessing remote folder: {str(metadata_error)}")
                                return
                        raise
                else:
                    result = self.dbx.files_list_folder_continue(cursor)
                
                for entry in result.entries:
                    if isinstance(entry, FileMetadata):
                        # Find the case-insensitive position of the remote folder in the path
                        entry_path = entry.path_display
                        if entry_path.lower().startswith(normalized_remote_folder.lower()):
                            # Replace the remote folder part while preserving the rest of the path
                            normalized_path = normalized_remote_folder + entry_path[len(normalized_remote_folder):]
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
        
        # Count local files
        for root, _, files in os.walk(self.local_folder):
            for file in files:
                local_path = Path(root) / file
                # Normalize the case of the path
                normalized_path = self._normalize_case(local_path, remote_files)
                if normalized_path is None:
                    # This is a new local file
                    normalized_path = local_path

                relative_path = local_path.relative_to(self.local_folder)
                # Skip files with forbidden characters
                if not self._validate_path(relative_path, os.sep):
                    continue
                dropbox_path = f"{normalized_remote_folder}/{str(relative_path).replace(os.sep, '/')}"
                if dropbox_path not in remote_files:
                    total_operations += 1

        # Count remaining remote files (files that exist only in Dropbox)
        for dropbox_path in remote_files:
            relative_path = dropbox_path[len(normalized_remote_folder):].lstrip('/')
            if not self._validate_path(relative_path, '/'):
                continue
            total_operations += 1

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
                        self._log_message(f"Skipped local file due to forbidden characters: {local_path}")
                        continue

                    # Get local file modification time and content hash
                    local_mtime = local_path.stat().st_mtime
                    local_hash = self._get_dropbox_content_hash(local_path)

                    if dropbox_path in remote_files:
                        # File exists in both places - compare content and timestamps
                        remote_entry = remote_files[dropbox_path]
                        remote_mtime = remote_entry.server_modified.timestamp()

                        if local_hash != remote_entry.content_hash:
                            if remote_mtime > local_mtime:
                                if download:
                                    # Remote is newer - download
                                    self._log_message(f"Downloading {dropbox_path} - Remote version newer")
                                    if dry_run:
                                        print(f"Downloading {dropbox_path} - Remote version newer")
                                    else:
                                        self.dbx.files_download_to_file(str(local_path), dropbox_path)
                            else:
                                if upload:
                                    # Local is newer - upload
                                    self._log_message(f"Uploading {local_path} - Local version newer")
                                    if dry_run:
                                        print(f"Uploading {local_path} - Local version newer")
                                    else:
                                        with open(local_path, 'rb') as f:
                                            self.dbx.files_upload(f.read(), dropbox_path, 
                                                                mode=dropbox.files.WriteMode.overwrite)
                        
                        # Remove from remote_files dict to track processed files
                        del remote_files[dropbox_path]
                    else:
                        if upload:
                            # File exists only locally - upload it
                            self._log_message(f"Uploading new file {local_path}")
                            if dry_run:
                                print(f"Uploading new file {local_path}")
                            else:
                                with open(local_path, 'rb') as f:
                                    self.dbx.files_upload(f.read(), dropbox_path)
                    
                    pbar.update(1)

            # Download remaining files that exist only in Dropbox
            for dropbox_path, entry in remote_files.items():
                relative_path = entry.path_display[len(normalized_remote_folder):].lstrip('/')
                local_path = self.local_folder / relative_path

                # Check for forbidden characters
                if not self._validate_path(relative_path, '/'):
                    self._log_message(f"Skipped remote file due to forbidden characters: {entry.path_display}")
                    continue

                if download:
                    # Create parent directories if needed
                    local_path.parent.mkdir(parents=True, exist_ok=True)

                    self._log_message(f"Downloading new file {dropbox_path}")
                    if dry_run:
                        print(f"Downloading new file {dropbox_path}")
                    else:
                        self.dbx.files_download_to_file(str(local_path), dropbox_path)
                pbar.update(1)

        print(f"Ended two-way synchronization between {self.remote_folder} and {self.local_folder}")
        self._log_message(f"Ended two-way synchronization between {self.remote_folder} and {self.local_folder}")


def main():
    parser = argparse.ArgumentParser(description='Dropbox Two-Way Sync Tool')
    parser.add_argument('--remote-folder', default='/', help='Dropbox folder path')
    parser.add_argument('--local-folder', default='D:/Dropbox', help='Local folder path')
    parser.add_argument('--auth-file', default='dropbox_auth.json', help='Authentication file path')
    parser.add_argument('--app-key', help='Dropbox app key (optional)')
    parser.add_argument('--app-secret', help='Dropbox app secret (optional)')
    parser.add_argument('--dry-run', action='store_true', help='Do not do any actual action, only write the log file of what would be done.')
    parser.add_argument('-D', '--download', action='store_true', help='Download from Dropbox.')
    parser.add_argument('-U', '--upload', action='store_true', help='Upload to Dropbox.')
    
    args = parser.parse_args()

    if not (args.download or args.upload):
        parser.error('No action requested, add -D and/or -U options')

    syncer = DropboxSyncer(
        auth_file=args.auth_file,
        remote_folder=args.remote_folder,
        local_folder=args.local_folder,
        app_key=args.app_key,
        app_secret=args.app_secret
    )
    syncer.sync(dry_run=args.dry_run, download=args.download, upload=args.upload)

if __name__ == "__main__":
    main()