# Saxo Bank API Authentication Guide

## Overview

The day2day application now provides a complete OAuth2 authentication flow for Saxo Bank API access. You no longer need to manually obtain access tokens - the system handles the entire authentication process.

## Quick Start

### 1. Set up your API credentials

Copy the example environment file:
```bash
cp .env.example .env
```

Edit `.env` and add your Saxo Bank API credentials:
```bash
SAXO_CLIENT_ID=your_client_id_here
SAXO_CLIENT_SECRET=your_client_secret_here
SAXO_REDIRECT_URI=https://jakobogtherese.dk
```

**Note**: You only need to set the CLIENT_ID, CLIENT_SECRET, and REDIRECT_URI. The access and refresh tokens will be obtained automatically.

### 2. Authenticate with Saxo Bank

Run the authentication command:
```bash
day2day auth
```

This will:
1. Generate an authorization URL
2. Open a browser session to Saxo Bank login
3. Guide you through the OAuth flow
4. Automatically save the access and refresh tokens

### 3. Start collecting data

Once authenticated, you can collect market data:
```bash
day2day collect --start-date 2023-01-01 --end-date 2023-12-31
```

## Authentication Features

### ✅ **Complete OAuth2 Flow**
- Interactive browser-based authentication
- Automatic authorization code exchange
- Token persistence to `.env` file
- Refresh token handling

### ✅ **Automatic Token Management**
- Checks token validity before API calls
- Automatic token refresh when expired
- Fallback to interactive auth if refresh fails
- Persistent token storage

### ✅ **User-Friendly Interface**
- Clear step-by-step instructions
- Visual confirmation of success/failure
- Error handling with helpful messages
- CLI commands for easy access

## Authentication Methods

### 1. Interactive CLI Authentication

```bash
# Start authentication process
day2day auth

# Check current authentication status
day2day auth --check
```

### 2. Programmatic Authentication

```python
from day2day.api.main import Day2DayAPI

api = Day2DayAPI()

# Check if authenticated
if api.check_authentication():
    print("Already authenticated")
else:
    # Perform authentication
    api.authenticate_saxo_bank()
```

### 3. Automatic Authentication

The system automatically handles authentication when you try to collect data:

```python
# This will automatically ensure valid authentication
api.collect_market_data("2023-01-01", "2023-12-31")
```

## Authentication Flow Detail

### Step 1: Authorization URL Generation
The system generates a Saxo Bank authorization URL with:
- Your client ID
- Redirect URI
- Required scopes (read, write)
- State parameter for security

### Step 2: User Authorization
You'll see output like:
```
============================================================
SAXO BANK API AUTHENTICATION
============================================================

1. Please open the following URL in your browser:

https://sim.logonvalidation.net/authorize?response_type=code&client_id=...

2. Log in to your Saxo Bank account
3. Authorize the application
4. Copy the authorization code from the callback URL

The callback URL will look like:
https://jakobogtherese.dk?code=AUTHORIZATION_CODE&state=day2day_auth

Copy the AUTHORIZATION_CODE part.
============================================================
```

### Step 3: Token Exchange
After you provide the authorization code, the system:
1. Exchanges the code for access and refresh tokens
2. Saves tokens to your `.env` file
3. Confirms successful authentication

### Step 4: Token Persistence
Tokens are saved in your `.env` file:
```bash
SAXO_ACCESS_TOKEN=eyJhbGciOiJFUzI1NiIsIng1dCI6...
SAXO_REFRESH_TOKEN=eyJhbGciOiJFUzI1NiIsIng1dCI6...
```

## Token Management

### Automatic Token Refresh
- System checks token validity before each API call
- Automatically refreshes expired tokens using refresh token
- Saves new tokens to `.env` file
- Transparent to the user

### Token Expiration Handling
If both access and refresh tokens are expired:
1. System detects invalid authentication
2. Prompts user for re-authentication
3. Starts interactive OAuth flow
4. Saves new tokens

### Manual Token Check
```bash
# Check if current tokens are valid
day2day auth --check

# Output examples:
# ✓ Authentication is valid
# ✗ Authentication is invalid or expired
```

## Error Handling

### Common Issues and Solutions

#### 1. Invalid Client Credentials
```
Error: Failed to obtain access token: 400 - Invalid client credentials
```
**Solution**: Check your `SAXO_CLIENT_ID` and `SAXO_CLIENT_SECRET` in `.env`

#### 2. Invalid Redirect URI
```
Error: redirect_uri_mismatch
```
**Solution**: Ensure `SAXO_REDIRECT_URI` matches your Saxo Bank app configuration

#### 3. Invalid Authorization Code
```
Error: Failed to obtain access token: 400 - Invalid authorization code
```
**Solution**: 
- Make sure you copied the complete authorization code
- Don't include extra characters or spaces
- Try the authentication process again

#### 4. Network/Connection Issues
```
Error: Request exception occurred: Connection timeout
```
**Solution**: 
- Check your internet connection
- Verify Saxo Bank services are available
- Try again later

### Debug Mode
Enable detailed logging for troubleshooting:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Security Considerations

### Token Storage
- Tokens are stored in `.env` file (gitignored)
- Never commit tokens to version control
- Use appropriate file permissions on `.env`

### Token Rotation
- Access tokens have limited lifetime (typically 1 hour)
- Refresh tokens have longer lifetime (typically 30 days)
- System automatically handles token rotation

### Network Security
- All API calls use HTTPS
- OAuth2 standard security practices
- State parameter prevents CSRF attacks

## Advanced Usage

### Custom Authentication Flow
For advanced use cases, you can access the authenticator directly:

```python
from day2day.data.auth import SaxoAuthenticator

auth = SaxoAuthenticator()

# Check token validity
if auth.is_token_valid():
    print("Token is valid")

# Force token refresh
auth.refresh_access_token_if_needed()

# Get authorization URL
auth_url = auth.build_authorization_url()
```

### Environment Variables
Override default settings with environment variables:
```bash
SAXO_BASE_URL=https://gateway.saxobank.com/sim/openapi
SAXO_AUTH_URL=https://sim.logonvalidation.net
```

## Migration from Manual Token Management

If you were previously setting `SAXO_ACCESS_TOKEN` manually:

1. **Remove manual token**: Delete `SAXO_ACCESS_TOKEN` from `.env`
2. **Run authentication**: Use `day2day auth` to get new tokens
3. **Verify**: Check that both access and refresh tokens are now in `.env`

## Troubleshooting

### Reset Authentication
To start fresh:
1. Remove `SAXO_ACCESS_TOKEN` and `SAXO_REFRESH_TOKEN` from `.env`
2. Run `day2day auth`
3. Complete the OAuth flow again

### Verify Configuration
Check your configuration:
```bash
day2day status
```

### Test API Access
Test that authentication works:
```bash
day2day auth --check
```

## Support

For authentication issues:
1. Check this guide first
2. Enable debug logging
3. Verify your Saxo Bank app configuration
4. Test with a minimal example
5. Check Saxo Bank API documentation

## Next Steps

Once authenticated, you can:
1. Collect market data: `day2day collect --start-date 2023-01-01 --end-date 2023-12-31`
2. Prepare training data: `day2day prepare --raw-data-file danish_stocks_1m.csv --output-title my_data --target-instrument NOVO-B.CO`
3. Train models: `day2day train --training-data-title my_data --target-instrument NOVO-B.CO`

The authentication system ensures you always have valid API access without manual token management!