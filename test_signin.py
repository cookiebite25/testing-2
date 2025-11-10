# test_signin.py
import os
from dotenv import load_dotenv
load_dotenv()
from supabase import create_client

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY")
print("SUPABASE_URL set:", bool(SUPABASE_URL))
print("SUPABASE_KEY set:", bool(SUPABASE_KEY))

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
email = "your-test-email@example.com"   # replace
password = "YourPassword"               # replace

try:
    resp = supabase.auth.sign_in_with_password({"email": email, "password": password})
    print("Raw response repr:", repr(resp))
    try:
        print("session (attr):", getattr(resp, 'session', None))
        print("user (attr):", getattr(resp, 'user', None))
    except Exception:
        pass
    if isinstance(resp, dict):
        print("response dict keys:", list(resp.keys()))
        print("session (dict):", resp.get('session'))
        print("user (dict):", resp.get('user'))
except Exception as e:
    print("Sign-in exception:", repr(e))