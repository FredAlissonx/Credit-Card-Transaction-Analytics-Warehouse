import os
import requests
from utils.config import session, logger

BASE_URL: str = "https://apisandbox.openbankproject.com/obp/v5.1.0"
ACCESS_TOKEN_ENV_VAR: str = "OBP_ACCESS_TOKEN"

def get_access_token() -> str:
    token = os.getenv(ACCESS_TOKEN_ENV_VAR)
    if not token:
        logger.error(f"{ACCESS_TOKEN_ENV_VAR} missing or empty.")
        raise EnvironmentError(f"{ACCESS_TOKEN_ENV_VAR} missing or empty")
    return token

def get_auth_headers(token: str) -> dict:
    """Construct authentication headers."""
    return {
        "Authorization": f"DirectLogin token={token}",
        "Content-Type": "application/json",
    }


def get_response(
    url: str,
    token: str,
    timeout: int = 10,
) -> requests.Response:
    """Perform a GET request and return the response, raising on HTTP errors."""
    try:
        resp = session.get(url, headers = get_auth_headers(token), timeout = timeout)
        resp.raise_for_status()
        logger.debug(f"API response status code: {resp.status_code}")
        return resp
    except requests.RequestException as err:
        logger.error(f"Error connecting to Open Bank Project API: {err}")
        raise


def get_banks_info() -> list[tuple[str, str]]:
    """Fetch banks and return a list of (id, full_name) tuples."""
    token = get_access_token()
    url = f"{BASE_URL}/banks"
    response = get_response(url, token)
    banks = response.json().get("banks", [])
    return [(bank.get("id"), bank.get("full_name")) for bank in banks]

if __name__ == "__main__":
    print(get_banks_info())
