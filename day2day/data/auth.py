"""Authentication module for Saxo Bank API."""

import requests
from urllib.parse import urlencode
from typing import Dict, Any
import logging
import time
from datetime import datetime, timedelta
from ..config.settings import settings

logger = logging.getLogger(__name__)

class SaxoAuthenticator:
    """Handles OAuth authentication with Saxo Bank API."""
    
    def __init__(self):
        self.client_id = settings.saxo_client_id
        self.client_secret = settings.saxo_client_secret
        self.redirect_uri = settings.saxo_redirect_uri
        self.auth_url = settings.saxo_auth_url
        self.access_token = settings.saxo_access_token
        self.token_expiry_time = None  # Will be set when token is obtained/refreshed
        self.refresh_token = settings.saxo_refresh_token
    
    def build_authorization_url(self, state: str = None, scope: str = None) -> str:
        """
        Constructs the authorization URL for Saxo Bank OAuth flow.
        
        Args:
            state: Optional state parameter for CSRF protection
            scope: Optional space-separated list of scopes
            
        Returns:
            Authorization URL string
        """
        base_url = f"{self.auth_url}/authorize"
        params = {
            "response_type": "code",
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
        }
        
        if state:
            params["state"] = state
        if scope:
            params["scope"] = scope
        
        return f"{base_url}?{urlencode(params)}"
    
    def exchange_code_for_token(self, authorization_code: str) -> Dict[str, Any]:
        """
        Exchanges authorization code for access token.
        
        Args:
            authorization_code: Authorization code from callback
            
        Returns:
            Token response dictionary
            
        Raises:
            Exception: If token exchange fails
        """
        token_url = f"{self.auth_url}/token"
        payload = {
            "grant_type": "authorization_code",
            "code": authorization_code,
            "redirect_uri": self.redirect_uri,
            "client_id": self.client_id,
            "client_secret": self.client_secret,
        }
        
        headers = {
            "Content-Type": "application/x-www-form-urlencoded"
        }
        
        response = requests.post(token_url, data=payload, headers=headers)
        
        if response.status_code not in [200, 201]:
            logger.error(f"Token exchange failed: {response.status_code} - {response.text}")
            raise Exception(f"Failed to obtain access token: {response.status_code} - {response.text}")
        
        token_data = response.json()
        
        # Store token expiry time if expires_in is provided
        if 'expires_in' in token_data:
            expires_in_seconds = token_data['expires_in']
            self.token_expiry_time = time.time() + expires_in_seconds
            logger.info(f"Token will expire in {expires_in_seconds} seconds")
        
        return token_data
    
    def refresh_token_request(self, refresh_token: str) -> Dict[str, Any]:
        """
        Refreshes access token using refresh token.
        
        Args:
            refresh_token: Refresh token from previous authentication
            
        Returns:
            New token response dictionary
            
        Raises:
            Exception: If token refresh fails
        """
        token_url = f"{self.auth_url}/token"
        payload = {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": self.client_id,
            "client_secret": self.client_secret,
        }
        
        headers = {
            "Content-Type": "application/x-www-form-urlencoded"
        }
        
        response = requests.post(token_url, data=payload, headers=headers)
        
        if response.status_code not in [200, 201]:
            logger.error(f"Token refresh failed: {response.status_code} - {response.text}")
            raise Exception(f"Failed to refresh access token: {response.status_code} - {response.text}")
        
        token_data = response.json()
        
        # Store token expiry time if expires_in is provided
        if 'expires_in' in token_data:
            expires_in_seconds = token_data['expires_in']
            self.token_expiry_time = time.time() + expires_in_seconds
            logger.info(f"Refreshed token will expire in {expires_in_seconds} seconds")
        
        return token_data
    
    def get_auth_headers(self) -> Dict[str, str]:
        """
        Get authentication headers for API requests.
        
        Returns:
            Headers dictionary with authorization
        """
        return {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
    
    def is_token_valid(self) -> bool:
        """
        Check if current access token is valid.
        
        Returns:
            True if token is valid, False otherwise
        """
        if not self.access_token:
            return False
        
        # Check if token is expired or will expire soon (within half the original lifetime)
        if self.token_expiry_time:
            current_time = time.time()
            if current_time >= self.token_expiry_time:
                logger.info("Token has expired based on stored expiry time")
                return False
        
        # Simple test call to check token validity
        try:
            headers = self.get_auth_headers()
            response = requests.get(
                f"{settings.saxo_base_url}/port/v1/users/me",
                headers=headers
            )
            return response.status_code == 200
        except Exception:
            return False
    
    def needs_refresh(self) -> bool:
        """
        Check if token needs to be refreshed (when half the time until expiration has passed).
        
        Returns:
            True if token should be refreshed, False otherwise
        """
        if not self.access_token or not self.token_expiry_time:
            return True
        
        current_time = time.time()
        
        # If token has expired, definitely needs refresh
        if current_time >= self.token_expiry_time:
            logger.info("Token has expired, needs refresh")
            return True
        
        # Check if we've passed the halfway point to expiration
        # We need to estimate the original token lifetime to calculate halfway point
        # Saxo tokens typically last about 1200 seconds (20 minutes)
        time_remaining = self.token_expiry_time - current_time
        
        # If less than half the typical token lifetime remains, refresh
        typical_lifetime = 1200  # 20 minutes in seconds
        if time_remaining < (typical_lifetime / 2):
            logger.info(f"Token has {time_remaining} seconds remaining, refreshing proactively")
            return True
        
        return False
    
    def get_access_token_interactive(self, save_to_env: bool = True) -> Dict[str, Any]:
        """
        Interactive OAuth flow to get access token.
        
        Args:
            save_to_env: Whether to save the token to environment/config
            
        Returns:
            Token response dictionary
        """
        logger.info("Starting interactive OAuth flow for Saxo Bank API")
        
        # Generate authorization URL
        auth_url = self.build_authorization_url(
            state="day2day_auth",
            scope="read write"
        )
        
        print("\n" + "="*60)
        print("SAXO BANK API AUTHENTICATION")
        print("="*60)
        print("\n1. Please open the following URL in your browser:")
        print(f"\n{auth_url}\n")
        print("2. Log in to your Saxo Bank account")
        print("3. Authorize the application")
        print("4. Copy the authorization code from the callback URL")
        print("\nThe callback URL will look like:")
        print(f"{self.redirect_uri}?code=AUTHORIZATION_CODE&state=day2day_auth")
        print("\nCopy the AUTHORIZATION_CODE part.")
        print("\n" + "="*60)
        
        # Get authorization code from user
        while True:
            auth_code = input("\nEnter the authorization code: ").strip()
            if auth_code:
                break
            print("Authorization code cannot be empty. Please try again.")
        
        # Exchange code for token
        try:
            token_response = self.exchange_code_for_token(auth_code)
            
            # Update instance with new token
            self.access_token = token_response.get("access_token")
            self.refresh_token = token_response.get("refresh_token")
            
            # Store expiry time
            if 'expires_in' in token_response:
                expires_in_seconds = token_response['expires_in']
                self.token_expiry_time = time.time() + expires_in_seconds
            
            if save_to_env:
                self._save_token_to_env(token_response)
            
            logger.info("Successfully obtained access token")
            print("\n✓ Authentication successful!")
            print(f"Access token expires in: {token_response.get('expires_in', 'unknown')} seconds")
            
            return token_response
            
        except Exception as e:
            logger.error(f"Failed to get access token: {e}")
            print(f"\n✗ Authentication failed: {e}")
            raise
    
    def _save_token_to_env(self, token_response: Dict[str, Any]) -> None:
        """
        Save tokens to .env file under the Saxo Bank API Configuration section.
        
        Args:
            token_response: Token response from OAuth flow
        """
        import os
        from pathlib import Path
        
        # Use MAIN_PATH if set, otherwise use project root
        main_path = os.getenv("MAIN_PATH")
        if main_path:
            env_file = Path(main_path).expanduser().resolve() / ".env"
        else:
            env_file = Path(__file__).parent.parent.parent / ".env"
        
        # Read existing .env file
        env_lines = []
        if env_file.exists():
            with open(env_file, 'r') as f:
                env_lines = f.readlines()
        
        # Update or add token entries
        access_token = token_response.get("access_token")
        refresh_token = token_response.get("refresh_token")
        
        # Remove existing token lines
        env_lines = [line for line in env_lines if not (
            line.startswith("SAXO_ACCESS_TOKEN=") or 
            line.startswith("SAXO_REFRESH_TOKEN=")
        )]
        
        # Find the position to insert tokens (after Saxo Bank API Configuration)
        insert_position = len(env_lines)  # Default to end if section not found
        
        for i, line in enumerate(env_lines):
            if line.strip() == "# Saxo Bank API Configuration":
                # Look for the next section or end of Saxo config
                for j in range(i + 1, len(env_lines)):
                    if env_lines[j].strip().startswith("#") and "Configuration" in env_lines[j]:
                        insert_position = j
                        break
                    elif j == len(env_lines) - 1:
                        insert_position = j + 1
                        break
                break
        
        # Insert tokens at the appropriate position
        token_lines = []
        if access_token:
            token_lines.append(f"SAXO_ACCESS_TOKEN={access_token}\n")
        if refresh_token:
            token_lines.append(f"SAXO_REFRESH_TOKEN={refresh_token}\n")
        
        # Insert the token lines
        for i, token_line in enumerate(token_lines):
            env_lines.insert(insert_position + i, token_line)
        
        # Write back to .env file
        with open(env_file, 'w') as f:
            f.writelines(env_lines)
        
        logger.info(f"Saved tokens to {env_file}")
        print(f"✓ Tokens saved to {env_file}")
    
    def refresh_access_token_if_needed(self) -> bool:
        """
        Refresh access token if it needs refresh (proactively at halfway point).
        
        Returns:
            True if token is valid (refreshed or already valid), False otherwise
        """
        if not self.needs_refresh():
            logger.debug("Access token is still valid and doesn't need refresh")
            return True
        
        # Try to refresh token
        refresh_token = self.refresh_token or settings.saxo_refresh_token
        
        if not refresh_token:
            logger.warning("No refresh token available, need to re-authenticate")
            return False
        
        try:
            logger.info("Refreshing access token proactively")
            token_response = self.refresh_token_request(refresh_token)
            
            # Update instance with new token
            self.access_token = token_response.get("access_token")
            if token_response.get("refresh_token"):
                self.refresh_token = token_response.get("refresh_token")
            
            # Save new tokens
            self._save_token_to_env(token_response)
            
            logger.info("Successfully refreshed access token")
            return True
            
        except Exception as e:
            logger.error(f"Failed to refresh access token: {e}")
            return False
    
    def ensure_valid_token(self) -> bool:
        """
        Ensure we have a valid access token, refreshing or re-authenticating if needed.
        
        Returns:
            True if valid token is available, False otherwise
        """
        # First check if current token is valid
        if self.is_token_valid():
            return True
        
        # Try to refresh token
        if self.refresh_access_token_if_needed():
            return True
        
        # If refresh failed, need interactive authentication
        logger.info("Token refresh failed, starting interactive authentication")
        print("\n⚠️  Access token is expired or invalid.")
        print("Starting interactive authentication process...")
        
        try:
            self.get_access_token_interactive()
            return True
        except Exception as e:
            logger.error(f"Interactive authentication failed: {e}")
            return False