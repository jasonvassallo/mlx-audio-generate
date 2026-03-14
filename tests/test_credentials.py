"""Tests for mlx_audiogen.credentials — Keychain-based credential manager."""

import pytest

from mlx_audiogen.credentials import CredentialManager

# ---------------------------------------------------------------------------
# Monkeypatched keyring: in-memory dict backend
# ---------------------------------------------------------------------------


@pytest.fixture()
def fake_keyring(monkeypatch):
    """Replace keyring get/set/delete with a plain dict."""
    store: dict[tuple[str, str], str] = {}

    def _get(service, username):
        return store.get((service, username))

    def _set(service, username, password):
        store[(service, username)] = password

    class _DeleteError(Exception):
        pass

    def _delete(service, username):
        if (service, username) not in store:
            raise _DeleteError("not found")
        del store[(service, username)]

    import mlx_audiogen.credentials as mod

    monkeypatch.setattr(mod.keyring, "get_password", _get)
    monkeypatch.setattr(mod.keyring, "set_password", _set)
    monkeypatch.setattr(mod.keyring, "delete_password", _delete)
    errors_ns = type("errors", (), {"PasswordDeleteError": _DeleteError})
    monkeypatch.setattr(mod.keyring, "errors", errors_ns)

    return store


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCredentialManager:
    def test_get_missing(self, fake_keyring):
        """get() returns None when no credential is stored and no env var."""
        cm = CredentialManager()
        assert cm.get("lastfm_api_key") is None

    def test_set_and_get(self, fake_keyring):
        """set() stores a credential, get() retrieves it."""
        cm = CredentialManager()
        cm.set("lastfm_api_key", "abc123")
        assert cm.get("lastfm_api_key") == "abc123"

    def test_delete(self, fake_keyring):
        """delete() removes a previously stored credential."""
        cm = CredentialManager()
        cm.set("discogs_token", "tok123")
        cm.delete("discogs_token")
        assert cm.get("discogs_token") is None

    def test_delete_missing_no_error(self, fake_keyring):
        """delete() on a missing credential does not raise."""
        cm = CredentialManager()
        cm.delete("lastfm_api_key")  # should not raise

    def test_env_var_fallback(self, fake_keyring, monkeypatch):
        """get() falls back to the environment variable when keychain is empty."""
        monkeypatch.setenv("LASTFM_API_KEY", "env_key_123")
        cm = CredentialManager()
        assert cm.get("lastfm_api_key") == "env_key_123"

    def test_keychain_priority_over_env(self, fake_keyring, monkeypatch):
        """Keychain value takes priority over env var."""
        monkeypatch.setenv("LASTFM_API_KEY", "env_key")
        cm = CredentialManager()
        cm.set("lastfm_api_key", "keychain_key")
        assert cm.get("lastfm_api_key") == "keychain_key"

    def test_status_empty(self, fake_keyring):
        """status() reports musicbrainz=True and others False when empty."""
        cm = CredentialManager()
        s = cm.status()
        assert s["musicbrainz"] is True
        assert s["lastfm"] is False
        assert s["discogs"] is False

    def test_status_after_set(self, fake_keyring):
        """status() reports True for services that have keys stored."""
        cm = CredentialManager()
        cm.set("lastfm_api_key", "key1")
        cm.set("discogs_token", "key2")
        s = cm.status()
        assert s["musicbrainz"] is True
        assert s["lastfm"] is True
        assert s["discogs"] is True

    def test_invalid_service(self, fake_keyring):
        """Operations on unknown service names raise ValueError."""
        cm = CredentialManager()
        with pytest.raises(ValueError, match="Unknown service"):
            cm.get("spotify_key")
        with pytest.raises(ValueError, match="Unknown service"):
            cm.set("spotify_key", "val")
        with pytest.raises(ValueError, match="Unknown service"):
            cm.delete("spotify_key")

    def test_masked_value(self, fake_keyring):
        """get_masked() shows only last 4 characters."""
        cm = CredentialManager()
        cm.set("lastfm_api_key", "abcdefgh1234")
        masked = cm.get_masked("lastfm_api_key")
        assert masked == "********1234"

    def test_masked_short_value(self, fake_keyring):
        """get_masked() on a short key (<=4 chars) masks entirely."""
        cm = CredentialManager()
        cm.set("lastfm_api_key", "ab")
        masked = cm.get_masked("lastfm_api_key")
        assert masked == "**"
