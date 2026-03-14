"""Keychain-based credential manager with environment variable fallback.

Stores and retrieves API keys for Last.fm and Discogs using the macOS
Keychain (via the ``keyring`` library).  Falls back to environment variables
when no Keychain entry exists.

MusicBrainz does not require an API key, so it is always reported as
available in :meth:`CredentialManager.status`.
"""

from __future__ import annotations

import os
from typing import Optional

import keyring

SERVICE_NAME = "mlx-audiogen"

# Maps logical service name -> environment variable fallback
_SERVICES: dict[str, str] = {
    "lastfm_api_key": "LASTFM_API_KEY",
    "discogs_token": "DISCOGS_TOKEN",
}


class CredentialManager:
    """Manage API credentials via macOS Keychain with env var fallback."""

    def _validate_service(self, service: str) -> None:
        """Raise ValueError if *service* is not a known service name."""
        if service not in _SERVICES:
            raise ValueError(
                f"Unknown service: {service!r}. "
                f"Valid services: {', '.join(sorted(_SERVICES))}"
            )

    def get(self, service: str) -> Optional[str]:
        """Retrieve a credential — Keychain first, then env var.

        Returns:
            The credential string, or ``None`` if not found anywhere.
        """
        self._validate_service(service)

        # Try keychain first
        value = keyring.get_password(SERVICE_NAME, service)
        if value is not None:
            return value

        # Fall back to environment variable
        return os.environ.get(_SERVICES[service])

    def set(self, service: str, value: str) -> None:
        """Store a credential in the Keychain."""
        self._validate_service(service)
        keyring.set_password(SERVICE_NAME, service, value)

    def delete(self, service: str) -> None:
        """Remove a credential from the Keychain (no-op if missing)."""
        self._validate_service(service)
        try:
            keyring.delete_password(SERVICE_NAME, service)
        except keyring.errors.PasswordDeleteError:
            pass

    def get_masked(self, service: str) -> Optional[str]:
        """Return the credential with all but the last 4 characters masked.

        Returns ``None`` if the credential is not stored.  If the credential
        is 4 characters or shorter, every character is masked.
        """
        value = self.get(service)
        if value is None:
            return None
        if len(value) <= 4:
            return "*" * len(value)
        return "*" * (len(value) - 4) + value[-4:]

    def status(self) -> dict[str, bool]:
        """Return availability status for each supported API.

        MusicBrainz is always ``True`` (no key required).  Last.fm and
        Discogs are ``True`` only when a key is available.
        """
        return {
            "musicbrainz": True,
            "lastfm": self.get("lastfm_api_key") is not None,
            "discogs": self.get("discogs_token") is not None,
        }
