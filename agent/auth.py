"""Globus token management for the ALCF inference endpoint (globus-sdk v4)."""
import json
import sys
import time
from pathlib import Path

GATEWAY_CLIENT_ID = "681c10cc-f684-4540-bcd7-0b4df3bc26ef"
AUTH_CLIENT_ID    = "58fdd3bc-e1c3-4ce5-80ea-8d6b87cfb944"
SCOPE             = f"https://auth.globus.org/scopes/{GATEWAY_CLIENT_ID}/action_all"
TOKEN_FILE        = Path.home() / ".globus" / "nebskill" / "tokens.json"


def _load_tokens() -> dict:
    if not TOKEN_FILE.exists():
        raise RuntimeError(
            "No ALCF token found. Authenticate first:\n"
            "  python agent/auth.py login"
        )
    return json.loads(TOKEN_FILE.read_text())


def _save_tokens(data: dict) -> None:
    TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    TOKEN_FILE.write_text(json.dumps(data, indent=2))


def get_access_token() -> str:
    """
    Return a valid access token for the ALCF inference endpoint.
    Auto-refreshes if the token is within 60 seconds of expiry.
    Raises RuntimeError if no token is cached or refresh fails.
    """
    import globus_sdk

    data = _load_tokens()

    # find the entry for the gateway resource server
    entry = data.get(GATEWAY_CLIENT_ID) or next(iter(data.values()), None)
    if not entry:
        raise RuntimeError("Token cache is empty. Re-authenticate:\n  python agent/auth.py login")

    expires_at = entry.get("expires_at_seconds", 0)
    if time.time() < expires_at - 60:
        return entry["access_token"]

    # token expired — try refresh
    refresh_token = entry.get("refresh_token")
    if not refresh_token:
        raise RuntimeError(
            "ALCF token expired and no refresh token stored.\n"
            "Re-authenticate:  python agent/auth.py login"
        )

    print("ALCF token expired — refreshing...", flush=True)
    client = globus_sdk.NativeAppAuthClient(AUTH_CLIENT_ID)
    authorizer = globus_sdk.RefreshTokenAuthorizer(
        refresh_token,
        client,
        access_token=entry["access_token"],
        expires_at=expires_at,
        on_refresh=lambda resp: _update_entry(data, resp),
    )
    # trigger a refresh by requesting the token
    token = authorizer.get_authorization_header().split(" ", 1)[1]
    return token


def _update_entry(data: dict, token_response) -> None:
    """Callback invoked by RefreshTokenAuthorizer on successful refresh."""
    by_rs = token_response.by_resource_server
    for rs_key, rs_data in by_rs.items():
        data[rs_key] = rs_data
    _save_tokens(data)


def login() -> None:
    """Interactive Globus login — run once to obtain and cache tokens."""
    import globus_sdk

    client = globus_sdk.NativeAppAuthClient(AUTH_CLIENT_ID)
    client.oauth2_start_flow(
        requested_scopes=SCOPE,
        redirect_uri="https://auth.globus.org/v2/web/auth-code",
        refresh_tokens=True,
    )

    url = client.oauth2_get_authorize_url()
    print(f"\nOpen this URL in a browser:\n\n  {url}\n")
    auth_code = input("Paste the auth code here: ").strip()

    tokens = client.oauth2_exchange_code_for_tokens(auth_code)
    data = dict(tokens.by_resource_server)
    _save_tokens(data)

    print(f"\nToken cached at {TOKEN_FILE}")
    # show which resource servers we got tokens for
    for rs, entry in data.items():
        exp = entry.get("expires_at_seconds", 0)
        exp_str = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(exp)) if exp else "unknown"
        print(f"  {rs}: expires {exp_str}")
    print("\nAuthentication complete.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "login":
        login()
    elif len(sys.argv) > 1 and sys.argv[1] == "check":
        try:
            tok = get_access_token()
            print(f"Token OK (first 12 chars: {tok[:12]}...)")
        except RuntimeError as e:
            print(f"ERROR: {e}")
    else:
        print("Usage:")
        print("  python agent/auth.py login   # authenticate")
        print("  python agent/auth.py check   # verify token")
