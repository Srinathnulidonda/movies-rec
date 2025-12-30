
import hashlib
import secrets
import urllib.parse

def generate_cinebrain_tokens():
    """Generate secure tokens for CineBrain operations"""
    
    # Generate a strong secret key
    operation_key = secrets.token_hex(32)
    
    # Generate the authentication token (matches server logic)
    auth_token = hashlib.sha256(operation_key.encode()).hexdigest()[:32]
    
    # Your app URL (replace with your actual URL)
    app_url = "https://cinebrain-server-1.onrender.com"
    
    # Generate complete UptimeRobot URLs
    basic_url = f"{app_url}/api/operation/refresh?token={auth_token}"
    custom_url = f"{app_url}/api/operation/refresh?token={auth_token}&force=false&personalized=true&user_limit=25"
    health_url = f"{app_url}/api/operations/health?token={auth_token}"
    
    print("🔐 CineBrain Security Configuration")
    print("=" * 50)
    print(f"Environment Variable:")
    print(f"OPERATION_KEY={operation_key}")
    print()
    print(f"UptimeRobot Token:")
    print(f"{auth_token}")
    print()
    print("📡 UptimeRobot Monitor URLs:")
    print(f"Basic Refresh: {basic_url}")
    print(f"Custom Refresh: {custom_url}")
    print(f"Health Check: {health_url}")
    print()
    print("🔒 HTTP Header Alternative:")
    print(f"URL: {app_url}/api/operation/refresh")
    print(f"Header: X-Task-Token: {auth_token}")

if __name__ == "__main__":
    generate_cinebrain_tokens()