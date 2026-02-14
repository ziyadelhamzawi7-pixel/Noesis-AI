"""
Google Drive API integration for browsing, listing, and downloading files.
"""

import os
import io
import requests as requests_lib
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from typing import Optional, List, Dict, Any
from pathlib import Path
from datetime import datetime

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request as GoogleAuthRequest
from loguru import logger

from app.config import settings
from app.google_oauth import google_oauth_service


# Supported file types and their MIME types
SUPPORTED_MIME_TYPES = {
    # PDF
    'application/pdf': '.pdf',
    # Microsoft Office
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document': '.docx',
    'application/msword': '.doc',
    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': '.xlsx',
    'application/vnd.ms-excel': '.xls',
    'application/vnd.openxmlformats-officedocument.presentationml.presentation': '.pptx',
    'application/vnd.ms-powerpoint': '.ppt',
    # CSV
    'text/csv': '.csv',
    'text/plain': '.txt',
    # Google Docs (will be exported)
    'application/vnd.google-apps.document': '.docx',
    'application/vnd.google-apps.spreadsheet': '.xlsx',
    'application/vnd.google-apps.presentation': '.pptx',
}

# Export formats for Google Docs
GOOGLE_EXPORT_FORMATS = {
    'application/vnd.google-apps.document': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    'application/vnd.google-apps.spreadsheet': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    'application/vnd.google-apps.presentation': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
}


class GoogleDriveService:
    """Service for interacting with Google Drive API."""

    def __init__(self, credentials: Credentials, timeout: int = 120):
        """
        Initialize Google Drive service.

        Args:
            credentials: Google OAuth credentials
            timeout: HTTP request timeout in seconds (default: 120 for large file downloads)
        """
        self.credentials = credentials
        self.timeout = timeout

        # Persistent session with connection pooling and transport-level retries
        # to avoid LibreSSL 2.8.3 TLS handshake failures on macOS
        self._session = requests_lib.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy, pool_maxsize=4)
        self._session.mount("https://", adapter)

        # Build service with credentials (uses default HTTP transport)
        # The requests library with google-auth handles SSL more reliably
        self.service = build('drive', 'v3', credentials=credentials)

    @classmethod
    def from_tokens(
        cls,
        access_token: str,
        refresh_token: Optional[str] = None,
        expires_at: Optional[str] = None,
        timeout: int = 120
    ) -> 'GoogleDriveService':
        """
        Create service from stored tokens.

        Args:
            access_token: Google access token
            refresh_token: Google refresh token
            expires_at: Token expiry datetime string
            timeout: HTTP request timeout in seconds

        Returns:
            GoogleDriveService instance
        """
        credentials = google_oauth_service.get_credentials_from_tokens(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_at=expires_at
        )
        return cls(credentials, timeout=timeout)

    def _resolve_shortcut(self, file_id: str) -> str:
        """
        If the file_id is a shortcut, return the target ID. Otherwise return the original ID.

        Args:
            file_id: Google Drive file ID (could be a shortcut or regular file/folder)

        Returns:
            The target ID if it's a shortcut, otherwise the original file_id
        """
        try:
            file = self.service.files().get(
                fileId=file_id,
                fields="mimeType, shortcutDetails",
                supportsAllDrives=True
            ).execute()

            if file.get('mimeType') == 'application/vnd.google-apps.shortcut':
                target_id = file.get('shortcutDetails', {}).get('targetId')
                if target_id:
                    logger.info(f"Resolved shortcut {file_id} to target {target_id}")
                    return target_id
            return file_id
        except Exception as e:
            logger.warning(f"Could not resolve shortcut {file_id}: {e}")
            return file_id

    def list_shared_drives(self, page_size: int = 100) -> List[Dict[str, Any]]:
        """
        List all Shared Drives (Team Drives) the user has access to.

        Args:
            page_size: Number of drives per page

        Returns:
            List of shared drives with id and name
        """
        try:
            drives = []
            page_token = None

            while True:
                result = self.service.drives().list(
                    pageSize=page_size,
                    pageToken=page_token,
                    fields="nextPageToken, drives(id, name)"
                ).execute()

                drives.extend(result.get('drives', []))
                page_token = result.get('nextPageToken')
                if not page_token:
                    break

            logger.info(f"Listed {len(drives)} shared drives")
            return drives

        except HttpError as e:
            logger.error(f"Failed to list shared drives: {e}")
            raise

    def list_files(
        self,
        folder_id: Optional[str] = None,
        page_size: int = 100,
        page_token: Optional[str] = None,
        include_folders: bool = True,
        view_mode: str = "my_drive",
        search_query: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        List files in a folder, shared files, or search results.

        Args:
            folder_id: Folder ID to list (None for root/My Drive, ignored for shared_with_me)
            page_size: Number of files per page
            page_token: Token for pagination
            include_folders: Whether to include folders in results
            view_mode: "my_drive" for user's files, "shared_with_me" for shared files
            search_query: Optional search term to filter by filename

        Returns:
            Dict with files list and pagination info
        """
        try:
            # Build query
            query_parts = []

            # Resolve folder_id if it's a shortcut (shortcuts point to another folder)
            resolved_folder_id = self._resolve_shortcut(folder_id) if folder_id else None

            if view_mode == "shared_drives":
                if resolved_folder_id:
                    # Navigating inside a Shared Drive - query by parent
                    query_parts.append(f"'{resolved_folder_id}' in parents")
                else:
                    # Root level: return the list of Shared Drives as folder entries
                    shared_drives = self.list_shared_drives()
                    drive_files = []
                    for drive in shared_drives:
                        drive_files.append({
                            'id': drive['id'],
                            'name': drive['name'],
                            'mimeType': 'application/vnd.google-apps.folder',
                            'isFolder': True,
                            'isSupported': True,
                            'size': None,
                            'modifiedTime': None,
                            'webViewLink': None,
                            'iconLink': None,
                            'extension': '',
                            'ownerEmail': None,
                            'sharedByEmail': None,
                            'shortcutTargetId': None
                        })
                    logger.info(f"Listed {len(drive_files)} shared drives as folders")
                    return {
                        'files': drive_files,
                        'nextPageToken': None,
                        'totalFiles': len(drive_files)
                    }
            elif view_mode == "shared_with_me":
                if resolved_folder_id:
                    # Navigating INTO a shared folder - query by parent
                    query_parts.append(f"'{resolved_folder_id}' in parents")
                else:
                    # Root of shared view - show all directly shared items
                    query_parts.append("sharedWithMe = true")
            else:
                # My Drive mode - browse by folder
                if resolved_folder_id:
                    query_parts.append(f"'{resolved_folder_id}' in parents")
                else:
                    query_parts.append("'root' in parents")

            # Add search filter if provided
            if search_query:
                # Escape single quotes in search term
                escaped_query = search_query.replace("'", "\\'")
                query_parts.append(f"name contains '{escaped_query}'")

            query_parts.append("trashed = false")

            query = " and ".join(query_parts)

            # Request fields - include owner info for shared files and shortcut details
            fields = "nextPageToken, files(id, name, mimeType, size, modifiedTime, parents, webViewLink, iconLink, owners, sharingUser, shortcutDetails)"

            result = self.service.files().list(
                q=query,
                pageSize=page_size,
                pageToken=page_token,
                fields=fields,
                orderBy="folder, name",
                supportsAllDrives=True,
                includeItemsFromAllDrives=True
            ).execute()

            files = result.get('files', [])

            # Categorize and enrich files
            processed_files = []
            for file in files:
                mime_type = file.get('mimeType', '')
                is_folder = mime_type == 'application/vnd.google-apps.folder'
                is_shortcut = mime_type == 'application/vnd.google-apps.shortcut'

                # Get the target ID if this is a shortcut to a folder
                shortcut_target_id = None
                if is_shortcut:
                    shortcut_details = file.get('shortcutDetails', {})
                    target_mime = shortcut_details.get('targetMimeType', '')
                    if target_mime == 'application/vnd.google-apps.folder':
                        is_folder = True  # Treat folder shortcuts as folders
                        shortcut_target_id = shortcut_details.get('targetId')

                if is_folder and not include_folders:
                    continue

                # Extract owner and sharing info
                owners = file.get('owners', [])
                owner_email = owners[0].get('emailAddress') if owners else None
                sharing_user = file.get('sharingUser', {})
                shared_by_email = sharing_user.get('emailAddress') if sharing_user else None

                file_info = {
                    'id': file['id'],
                    'name': file['name'],
                    'mimeType': mime_type,
                    'isFolder': is_folder,
                    'isSupported': mime_type in SUPPORTED_MIME_TYPES or is_folder,
                    'size': int(file.get('size', 0)) if file.get('size') else None,
                    'modifiedTime': file.get('modifiedTime'),
                    'webViewLink': file.get('webViewLink'),
                    'iconLink': file.get('iconLink'),
                    'extension': SUPPORTED_MIME_TYPES.get(mime_type, ''),
                    'ownerEmail': owner_email,
                    'sharedByEmail': shared_by_email,
                    'shortcutTargetId': shortcut_target_id
                }
                processed_files.append(file_info)

            log_context = f"view_mode={view_mode}"
            if search_query:
                log_context += f", search='{search_query}'"
            if folder_id and view_mode == "my_drive":
                log_context += f", folder={folder_id}"
            logger.info(f"Listed {len(processed_files)} files ({log_context})")

            return {
                'files': processed_files,
                'nextPageToken': result.get('nextPageToken'),
                'totalFiles': len(processed_files)
            }

        except HttpError as e:
            logger.error(f"Google Drive API error: {e}")
            raise

    def get_folder_path(self, folder_id: str) -> List[Dict[str, str]]:
        """
        Get the full path (breadcrumb) of a folder.

        Args:
            folder_id: Folder ID

        Returns:
            List of folder info from root to current folder
        """
        try:
            path = []
            current_id = folder_id

            while current_id:
                file = self.service.files().get(
                    fileId=current_id,
                    fields="id, name, parents"
                ).execute()

                path.insert(0, {
                    'id': file['id'],
                    'name': file['name']
                })

                parents = file.get('parents', [])
                current_id = parents[0] if parents else None

            return path

        except HttpError as e:
            logger.error(f"Failed to get folder path: {e}")
            return []

    def get_file_info(self, file_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a file.

        Args:
            file_id: Google Drive file ID

        Returns:
            File metadata
        """
        try:
            file = self.service.files().get(
                fileId=file_id,
                fields="id, name, mimeType, size, modifiedTime, createdTime, parents, webViewLink, description"
            ).execute()

            mime_type = file.get('mimeType', '')

            return {
                'id': file['id'],
                'name': file['name'],
                'mimeType': mime_type,
                'isFolder': mime_type == 'application/vnd.google-apps.folder',
                'isSupported': mime_type in SUPPORTED_MIME_TYPES,
                'size': int(file.get('size', 0)) if file.get('size') else None,
                'modifiedTime': file.get('modifiedTime'),
                'createdTime': file.get('createdTime'),
                'webViewLink': file.get('webViewLink'),
                'description': file.get('description'),
                'extension': SUPPORTED_MIME_TYPES.get(mime_type, '')
            }

        except HttpError as e:
            logger.error(f"Failed to get file info: {e}")
            raise

    def download_file(self, file_id: str, destination_path: str) -> str:
        """
        Download a file from Google Drive.

        Args:
            file_id: Google Drive file ID
            destination_path: Local path to save the file

        Returns:
            Path to downloaded file
        """
        try:
            # Get file info first
            file_info = self.get_file_info(file_id)
            mime_type = file_info['mimeType']
            file_name = file_info['name']

            # Create destination directory
            dest_path = Path(destination_path)
            dest_path.parent.mkdir(parents=True, exist_ok=True)

            # Refresh credentials to get a valid access token
            if self.credentials.expired or not self.credentials.token:
                self.credentials.refresh(GoogleAuthRequest())

            # Use requests library for downloads to bypass LibreSSL 2.8.3 SSL issues
            # (httplib2/MediaIoBaseDownload fails with SSL errors on macOS system Python)
            headers = {'Authorization': f'Bearer {self.credentials.token}'}

            if mime_type in GOOGLE_EXPORT_FORMATS:
                export_mime = GOOGLE_EXPORT_FORMATS[mime_type]
                extension = SUPPORTED_MIME_TYPES[mime_type]

                if not dest_path.suffix:
                    dest_path = dest_path.with_suffix(extension)

                url = f'https://www.googleapis.com/drive/v3/files/{file_id}/export?mimeType={export_mime}'
            else:
                url = f'https://www.googleapis.com/drive/v3/files/{file_id}?alt=media'

            response = self._session.get(url, headers=headers, stream=True, timeout=self.timeout)
            response.raise_for_status()

            with open(str(dest_path), 'wb') as f:
                for chunk in response.iter_content(chunk_size=settings.drive_download_chunk_size):
                    f.write(chunk)

            logger.info(f"Downloaded file: {file_name} -> {dest_path}")

            return str(dest_path)

        except (HttpError, requests_lib.RequestException) as e:
            logger.error(f"Failed to download file: {e}")
            raise

    def download_file_to_bytes(self, file_id: str) -> Dict[str, Any]:
        """
        Download a file from Google Drive to memory.

        Args:
            file_id: Google Drive file ID

        Returns:
            Dict with 'content' (bytes), 'filename', 'mimeType', 'size'
        """
        try:
            # Get file info first
            file_info = self.get_file_info(file_id)
            mime_type = file_info['mimeType']
            file_name = file_info['name']
            output_mime_type = mime_type

            # Refresh credentials to get a valid access token
            if self.credentials.expired or not self.credentials.token:
                self.credentials.refresh(GoogleAuthRequest())

            headers = {'Authorization': f'Bearer {self.credentials.token}'}

            # Check if it's a Google Docs file that needs export
            if mime_type in GOOGLE_EXPORT_FORMATS:
                export_mime = GOOGLE_EXPORT_FORMATS[mime_type]
                extension = SUPPORTED_MIME_TYPES[mime_type]
                output_mime_type = export_mime

                if not file_name.endswith(extension):
                    file_name = file_name + extension

                url = f'https://www.googleapis.com/drive/v3/files/{file_id}/export?mimeType={export_mime}'
            else:
                url = f'https://www.googleapis.com/drive/v3/files/{file_id}?alt=media'

            # Use persistent session to bypass LibreSSL 2.8.3 SSL issues
            response = self._session.get(url, headers=headers, timeout=self.timeout)
            response.raise_for_status()

            content = response.content
            logger.info(f"Downloaded file to memory: {file_name} ({len(content)} bytes)")

            return {
                'content': content,
                'filename': file_name,
                'mimeType': output_mime_type,
                'size': len(content)
            }

        except (HttpError, requests_lib.RequestException) as e:
            logger.error(f"Failed to download file to bytes: {e}")
            raise

    def download_file_to_disk(self, file_id: str, destination_path: str, chunk_size: int = settings.drive_download_chunk_size) -> Dict[str, Any]:
        """
        Stream-download a file from Google Drive to disk, returning metadata.
        Unlike download_file_to_bytes, this never loads the full file into memory.

        Args:
            file_id: Google Drive file ID
            destination_path: Local path to save the file
            chunk_size: Size of each streamed chunk in bytes

        Returns:
            Dict with 'filename', 'mimeType', 'size', 'path'
        """
        try:
            file_info = self.get_file_info(file_id)
            mime_type = file_info['mimeType']
            file_name = file_info['name']
            output_mime_type = mime_type

            if self.credentials.expired or not self.credentials.token:
                self.credentials.refresh(GoogleAuthRequest())

            headers = {'Authorization': f'Bearer {self.credentials.token}'}

            if mime_type in GOOGLE_EXPORT_FORMATS:
                export_mime = GOOGLE_EXPORT_FORMATS[mime_type]
                extension = SUPPORTED_MIME_TYPES[mime_type]
                output_mime_type = export_mime
                if not file_name.endswith(extension):
                    file_name = file_name + extension
                url = f'https://www.googleapis.com/drive/v3/files/{file_id}/export?mimeType={export_mime}'
            else:
                url = f'https://www.googleapis.com/drive/v3/files/{file_id}?alt=media'

            dest = Path(destination_path)
            dest.parent.mkdir(parents=True, exist_ok=True)

            response = self._session.get(url, headers=headers, stream=True, timeout=self.timeout)
            response.raise_for_status()

            total_bytes = 0
            with open(str(dest), 'wb') as f:
                for data in response.iter_content(chunk_size=chunk_size):
                    f.write(data)
                    total_bytes += len(data)

            logger.info(f"Stream-downloaded file: {file_name} ({total_bytes} bytes) -> {dest}")

            return {
                'filename': file_name,
                'mimeType': output_mime_type,
                'size': total_bytes,
                'path': str(dest)
            }

        except (HttpError, requests_lib.RequestException) as e:
            logger.error(f"Failed to stream-download file: {e}")
            raise

    def list_all_files_in_folder(
        self,
        folder_id: str,
        recursive: bool = True,
        supported_only: bool = True
    ) -> List[Dict[str, Any]]:
        """
        List all files in a folder (optionally recursive).

        Args:
            folder_id: Folder ID to scan
            recursive: Whether to include subfolders
            supported_only: Only return supported file types

        Returns:
            List of all files
        """
        all_files = []

        def scan_folder(fid: str, path: str = ""):
            result = self.list_files(folder_id=fid, include_folders=True)

            for file in result['files']:
                file_path = f"{path}/{file['name']}" if path else file['name']
                file['path'] = file_path

                if file['isFolder']:
                    if recursive:
                        scan_folder(file['id'], file_path)
                else:
                    if not supported_only or file['isSupported']:
                        all_files.append(file)

            # Handle pagination
            while result.get('nextPageToken'):
                result = self.list_files(
                    folder_id=fid,
                    page_token=result['nextPageToken'],
                    include_folders=True
                )
                for file in result['files']:
                    file_path = f"{path}/{file['name']}" if path else file['name']
                    file['path'] = file_path

                    if file['isFolder']:
                        if recursive:
                            scan_folder(file['id'], file_path)
                    else:
                        if not supported_only or file['isSupported']:
                            all_files.append(file)

        scan_folder(folder_id)

        logger.info(f"Found {len(all_files)} files in folder {folder_id}")
        return all_files

    def list_folder_contents_only(
        self,
        folder_id: str,
        supported_only: bool = True
    ) -> Dict[str, Any]:
        """
        List immediate contents of a folder without recursion or downloading.
        Used for discovery phase to count files per folder.

        Args:
            folder_id: Folder ID to scan
            supported_only: Only count supported file types

        Returns:
            Dict with:
            - subfolders: List of immediate subfolders [{id, name}]
            - files: List of files [{id, name, mimeType, size, modifiedTime}]
            - file_count: Number of (supported) files
            - total_size: Total size of files in bytes
        """
        subfolders = []
        files = []
        skipped_files = []
        total_size = 0

        # Extension-based fallback for files with generic/wrong MIME types
        SUPPORTED_EXTENSIONS = {'.pdf', '.docx', '.doc', '.xlsx', '.xls', '.pptx', '.ppt', '.csv', '.txt'}

        # Resolve folder_id if it's a shortcut
        resolved_folder_id = self._resolve_shortcut(folder_id)

        page_token = None
        while True:
            try:
                query = f"'{resolved_folder_id}' in parents and trashed = false"
                fields = "nextPageToken, files(id, name, mimeType, size, modifiedTime, shortcutDetails)"

                result = self.service.files().list(
                    q=query,
                    pageSize=1000,
                    pageToken=page_token,
                    fields=fields,
                    supportsAllDrives=True,
                    includeItemsFromAllDrives=True
                ).execute()

                for file in result.get('files', []):
                    mime_type = file.get('mimeType', '')
                    is_folder = mime_type == 'application/vnd.google-apps.folder'
                    is_shortcut = mime_type == 'application/vnd.google-apps.shortcut'

                    # Handle shortcuts to folders
                    shortcut_target_id = None
                    if is_shortcut:
                        shortcut_details = file.get('shortcutDetails', {})
                        target_mime = shortcut_details.get('targetMimeType', '')
                        if target_mime == 'application/vnd.google-apps.folder':
                            is_folder = True
                            shortcut_target_id = shortcut_details.get('targetId')

                    if is_folder:
                        subfolders.append({
                            'id': shortcut_target_id or file['id'],
                            'name': file['name']
                        })
                    else:
                        # Check if file type is supported (by MIME type or file extension)
                        file_ext = os.path.splitext(file['name'])[1].lower()
                        is_supported = mime_type in SUPPORTED_MIME_TYPES or file_ext in SUPPORTED_EXTENSIONS
                        if not supported_only or is_supported:
                            file_size = int(file.get('size', 0)) if file.get('size') else 0
                            files.append({
                                'id': file['id'],
                                'name': file['name'],
                                'mimeType': mime_type,
                                'size': file_size,
                                'modifiedTime': file.get('modifiedTime')
                            })
                            total_size += file_size
                        elif supported_only:
                            skipped_files.append({'name': file['name'], 'mimeType': mime_type})
                            logger.debug(f"Skipping unsupported file: {file['name']} (mime={mime_type})")

                page_token = result.get('nextPageToken')
                if not page_token:
                    break

            except HttpError as e:
                logger.error(f"Failed to list folder contents: {e}")
                raise

        return {
            'subfolders': subfolders,
            'files': files,
            'file_count': len(files),
            'total_size': total_size,
            'skipped_files': skipped_files,
            'skipped_count': len(skipped_files),
        }

    def watch_folder(self, folder_id: str, webhook_url: str) -> Dict[str, Any]:
        """
        Set up a watch on a folder for changes (for auto-sync).

        Note: This requires a publicly accessible webhook URL.
        For local development, consider polling instead.

        Args:
            folder_id: Folder to watch
            webhook_url: URL to receive notifications

        Returns:
            Watch channel info
        """
        import uuid

        try:
            channel_id = str(uuid.uuid4())

            body = {
                'id': channel_id,
                'type': 'web_hook',
                'address': webhook_url,
                'expiration': int((datetime.utcnow().timestamp() + 86400) * 1000)  # 24 hours
            }

            result = self.service.files().watch(
                fileId=folder_id,
                body=body
            ).execute()

            logger.info(f"Created watch channel {channel_id} for folder {folder_id}")

            return {
                'channelId': result.get('id'),
                'resourceId': result.get('resourceId'),
                'expiration': result.get('expiration')
            }

        except HttpError as e:
            logger.error(f"Failed to create watch: {e}")
            raise

    def get_changes(self, start_page_token: str) -> Dict[str, Any]:
        """
        Get changes since a specific page token (for polling-based sync).

        Args:
            start_page_token: Token from previous call or initial token

        Returns:
            Dict with changes and new page token
        """
        try:
            changes = []
            page_token = start_page_token

            while page_token:
                result = self.service.changes().list(
                    pageToken=page_token,
                    spaces='drive',
                    fields='nextPageToken, newStartPageToken, changes(fileId, removed, file(id, name, mimeType, parents))'
                ).execute()

                changes.extend(result.get('changes', []))
                page_token = result.get('nextPageToken')

            new_start_token = result.get('newStartPageToken', start_page_token)

            logger.info(f"Found {len(changes)} changes")

            return {
                'changes': changes,
                'newStartPageToken': new_start_token
            }

        except HttpError as e:
            logger.error(f"Failed to get changes: {e}")
            raise

    def get_start_page_token(self) -> str:
        """
        Get the initial page token for change tracking.

        Returns:
            Start page token
        """
        try:
            result = self.service.changes().getStartPageToken().execute()
            return result.get('startPageToken')
        except HttpError as e:
            logger.error(f"Failed to get start page token: {e}")
            raise
