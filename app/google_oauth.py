"""
Google OAuth 2.0 integration for Google Drive access.
Handles authentication flow, token management, and refresh.
"""

import os
import json
import secrets
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from pathlib import Path

# Allow OAuth scope changes (e.g. user declines Drive access but grants profile/email)
os.environ["OAUTHLIB_RELAX_TOKEN_SCOPE"] = "1"

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from google.auth.transport.requests import Request
from loguru import logger

from app.config import settings

# OAuth 2.0 scopes for Google Drive
SCOPES = [
    'https://www.googleapis.com/auth/drive.readonly',  # Read-only access to Drive
    'https://www.googleapis.com/auth/drive.file',  # Create/edit files the app creates (memo export)
    'https://www.googleapis.com/auth/userinfo.email',  # Get user email
    'https://www.googleapis.com/auth/userinfo.profile',  # Get user profile
    'openid'  # OpenID Connect
]

# OAuth client configuration
CLIENT_CONFIG = {
    "web": {
        "client_id": settings.google_client_id,
        "client_secret": settings.google_client_secret,
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "redirect_uris": ["http://localhost:8000/api/auth/google/callback"],
        "javascript_origins": ["http://localhost:3000", "http://localhost:5173"]
    }
}


class GoogleOAuthService:
    """Service for handling Google OAuth authentication."""

    def __init__(self):
        self.client_config = CLIENT_CONFIG
        self._validate_config()

    def _validate_config(self):
        """Validate OAuth configuration."""
        if not settings.google_client_id or not settings.google_client_secret:
            logger.warning("Google OAuth credentials not configured. Set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET in .env")

    def create_auth_url(self, redirect_uri: str, state: Optional[str] = None) -> Dict[str, str]:
        """
        Create Google OAuth authorization URL.

        Args:
            redirect_uri: Callback URL after authorization
            state: Optional state parameter for CSRF protection

        Returns:
            Dict with auth_url and state
        """
        if not state:
            state = secrets.token_urlsafe(32)

        # Update redirect URI in config
        config = self.client_config.copy()
        config["web"]["redirect_uris"] = [redirect_uri]

        flow = Flow.from_client_config(
            config,
            scopes=SCOPES,
            redirect_uri=redirect_uri
        )

        auth_url, _ = flow.authorization_url(
            access_type='offline',  # Get refresh token
            include_granted_scopes='true',
            prompt='consent',  # Always show consent screen to get refresh token
            state=state
        )

        logger.info(f"Created OAuth authorization URL with state: {state[:8]}...")
        logger.info(f"Full auth URL: {auth_url}")

        return {
            "auth_url": auth_url,
            "state": state
        }

    def exchange_code_for_tokens(self, code: str, redirect_uri: str) -> Dict[str, Any]:
        """
        Exchange authorization code for access and refresh tokens.

        Args:
            code: Authorization code from Google
            redirect_uri: Same redirect URI used in authorization

        Returns:
            Dict with tokens and user info
        """
        # Update redirect URI in config
        config = self.client_config.copy()
        config["web"]["redirect_uris"] = [redirect_uri]

        flow = Flow.from_client_config(
            config,
            scopes=SCOPES,
            redirect_uri=redirect_uri
        )

        # Exchange code for tokens
        flow.fetch_token(code=code)
        credentials = flow.credentials

        # Get user info
        user_info = self._get_user_info(credentials)

        # Calculate token expiry
        expires_at = None
        if credentials.expiry:
            expires_at = credentials.expiry.isoformat()

        logger.info(f"Successfully exchanged code for tokens for user: {user_info.get('email')}")

        return {
            "access_token": credentials.token,
            "refresh_token": credentials.refresh_token,
            "token_type": "Bearer",
            "expires_at": expires_at,
            "scopes": list(credentials.scopes) if credentials.scopes else SCOPES,
            "user_info": user_info
        }

    def _get_user_info(self, credentials: Credentials) -> Dict[str, Any]:
        """
        Get user information from Google.

        Args:
            credentials: Google OAuth credentials

        Returns:
            Dict with user info (email, name, picture)
        """
        import httpx

        headers = {"Authorization": f"Bearer {credentials.token}"}

        response = httpx.get(
            "https://www.googleapis.com/oauth2/v2/userinfo",
            headers=headers
        )

        if response.status_code == 200:
            return response.json()
        else:
            logger.warning(f"Failed to get user info: {response.status_code}")
            return {}

    def refresh_access_token(self, refresh_token: str) -> Optional[Dict[str, Any]]:
        """
        Refresh an expired access token.

        Args:
            refresh_token: Google refresh token

        Returns:
            Dict with new access token and expiry, or None if failed
        """
        try:
            credentials = Credentials(
                token=None,
                refresh_token=refresh_token,
                token_uri="https://oauth2.googleapis.com/token",
                client_id=settings.google_client_id,
                client_secret=settings.google_client_secret,
            )

            # Refresh the token
            credentials.refresh(Request())

            expires_at = None
            if credentials.expiry:
                expires_at = credentials.expiry.isoformat()

            logger.info("Successfully refreshed access token")

            return {
                "access_token": credentials.token,
                "expires_at": expires_at
            }

        except Exception as e:
            logger.error(f"Failed to refresh token: {e}")
            return None

    def revoke_token(self, token: str) -> bool:
        """
        Revoke a Google OAuth token.

        Args:
            token: Access or refresh token to revoke

        Returns:
            True if revoked successfully
        """
        import httpx

        try:
            response = httpx.post(
                "https://oauth2.googleapis.com/revoke",
                params={"token": token},
                headers={"Content-Type": "application/x-www-form-urlencoded"}
            )

            if response.status_code == 200:
                logger.info("Token revoked successfully")
                return True
            else:
                logger.warning(f"Token revocation failed: {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"Failed to revoke token: {e}")
            return False

    def get_credentials_from_tokens(
        self,
        access_token: str,
        refresh_token: Optional[str] = None,
        expires_at: Optional[str] = None
    ) -> Credentials:
        """
        Create Credentials object from stored tokens.

        Args:
            access_token: Google access token
            refresh_token: Google refresh token (optional)
            expires_at: Token expiry datetime string (optional)

        Returns:
            Google Credentials object
        """
        expiry = None
        if expires_at:
            if isinstance(expires_at, datetime):
                expiry = expires_at
            else:
                expiry = datetime.fromisoformat(expires_at.replace('Z', '+00:00'))

        credentials = Credentials(
            token=access_token,
            refresh_token=refresh_token,
            token_uri="https://oauth2.googleapis.com/token",
            client_id=settings.google_client_id,
            client_secret=settings.google_client_secret,
            expiry=expiry
        )

        return credentials


# Global service instance
google_oauth_service = GoogleOAuthService()
