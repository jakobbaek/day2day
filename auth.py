import requests
from urllib.parse import urlencode

def build_authorization_url(client_id, redirect_uri, state=None, scope=None):
    """
    Constructs the authorization URL that directs a user to Saxo Bank’s login page
    using the endpoint provided by Saxo Bank for developers.

    Parameters:
        client_id (str): Your application's client ID.
        redirect_uri (str): The callback URL registered with Saxo Bank.
        state (str, optional): A unique string to mitigate CSRF attacks.
        scope (str, optional): A space-separated list of scopes to request.

    Returns:
        str: The full authorization URL.
    """
    base_url = "https://sim.logonvalidation.net/authorize"
    params = {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": redirect_uri,
    }
    if state:
        params["state"] = state
    if scope:
        params["scope"] = scope

    return f"{base_url}?{urlencode(params)}"


def get_access_token(client_id, client_secret, redirect_uri, authorization_code):
    """
    Exchanges an authorization code for an access token using the Saxo Bank token endpoint.

    Parameters:
        client_id (str): Your application's client ID.
        client_secret (str): Your application's client secret.
        redirect_uri (str): The callback URL registered with Saxo Bank.
        authorization_code (str): The code received after user authentication.

    Returns:
        dict: The JSON response containing the access token, refresh token, etc.
    """
    token_url = "https://sim.logonvalidation.net/token"
    payload = {
        "grant_type": "authorization_code",
        "code": authorization_code,
        "redirect_uri": redirect_uri,
        "client_id": client_id,
        "client_secret": client_secret,
    }
    headers = {
        "Content-Type": "application/x-www-form-urlencoded"
    }

    response = requests.post(token_url, data=payload, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Failed to obtain access token: {response.status_code} - {response.text}")
    
    return response.json()


if __name__ == "__main__":
    # Replace these placeholders with your application's details.
    CLIENT_ID = "960be823ff96452f8447b3d8df4722ee"
    CLIENT_SECRET = "fd284541010e4bf9ac56a217fad39835"
    REDIRECT_URI = "https://jakobogtherese.dk"
    STATE = "YOUR_RANDOM_STATE"       # Optional: for CSRF protection.
    SCOPE = "read write"              # Optional: adjust based on your app's needs.

    # Step 1: Generate the authorization URL and direct your user there.
    auth_url = build_authorization_url(CLIENT_ID, REDIRECT_URI, state=STATE, scope=SCOPE)
    print("Please navigate to the following URL to authorize the application:")
    print(auth_url)

    # After authorization, Saxo Bank redirects to your redirect_uri with a 'code' parameter.
    authorization_code = input("Enter the authorization code received: ").strip()

    # Step 2: Exchange the authorization code for an access token.
    try:
        tokens = get_access_token(CLIENT_ID, CLIENT_SECRET, REDIRECT_URI, authorization_code)
        print("Access Token:", tokens.get("access_token"))
        print("Refresh Token:", tokens.get("refresh_token"))
    except Exception as e:
        print("An error occurred during token exchange:", e)
