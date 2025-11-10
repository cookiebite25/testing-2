#from ipaddress import ip_address
import os
import time
import logging
import json
import hashlib
import requests
from flask import Flask, request, render_template, jsonify, send_file, Response, send_from_directory, redirect, url_for, flash, session
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
from PIL import Image, ImageDraw
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas as pdf_canvas
from reportlab.lib.utils import ImageReader
from io import BytesIO
import re
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import base64
from functools import wraps
from pathlib import Path
import random
import asyncio
import inspect



def _ensure_placeholder():
    path = Path("static/placeholder.png")
    if path.exists():
        return
    img = Image.new("RGB", (600, 400), "#2c2c2c")
    d = ImageDraw.Draw(img)
    # ImageDraw.text does not accept font_size; use a default font if available
    try:
        from PIL import ImageFont
        font = ImageFont.load_default()
    except Exception:
        font = None
    d.text((50, 170), "Image not available", fill="#bbbbbb", font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)

# Ensure placeholder exists (do not use logger here because logging is configured later)
_ensure_placeholder()


def get_simple_fallback_response(user_message):
    """Provide simple cached responses for common questions when AI is down."""
    user_lower = user_message.lower().strip()
    
    SIMPLE_RESPONSES = {
        "hello": "Hello! I'm your AI network assistant. How can I help with your network planning today?",
        "hi": "Hi there! I specialize in network architecture and router placement. What can I help you with?",
        "help": "I can help you with: router placement, WiFi optimization, network budgets, signal coverage, and equipment recommendations. Just ask!",
        "how are you": "I'm functioning well! Ready to help you design the perfect network setup. What's your network challenge?",
        "how does network architecture work": "Network architecture involves designing the structure of your network - including router placement, cable routing, and device connectivity. When the AI service is available, I can analyze your specific blueprint to create an optimal layout.",
        
        "router placement": "Optimal router placement considers room layout, wall materials, and device locations. Upload a blueprint when the AI is available for personalized placement recommendations.",
        
        "wifi coverage": "Good WiFi coverage depends on router positioning, avoiding signal obstructions, and proper equipment selection. I can provide a signal heatmap analysis once the AI service recovers.",
        
        "network budget": "Typical home network budgets start around R2,500. For detailed cost breakdowns based on your specific needs, please try again when the AI service is fully operational."

    }
    
    # Exact matches
    if user_lower in SIMPLE_RESPONSES:
        return SIMPLE_RESPONSES[user_lower]
    
    # Partial matches
    if any(word in user_lower for word in ['router', 'wifi', 'network', 'signal', 'coverage']):
        return "I'd normally provide detailed network advice, but the AI service is temporarily busy. Please try again in a moment for specific router placement and network optimization help."
    
    if any(word in user_lower for word in ['budget', 'cost', 'price', 'money']):
        return "For budget planning, I typically recommend starting with R2,500 for basic setups. The AI service will be back shortly to provide detailed cost analysis."
    
    return "I'm currently experiencing high demand. Please try again in a few moments for detailed network architecture assistance."

# Import Supabase helper functions
from supabase_client import (
    supabase,
    redis_client, 
    REDIS_AVAILABLE,
    store_config, 
    get_config, 
    store_suggestion, 
    get_suggestion,
    store_chat_cache,
    get_chat_cache,
    clear_all_cache,
    clear_redis_only,  
    log_user_event,
    log_visitor,
    check_redis_health,  
    check_supabase_health, 
    get_storage_status 
)

def refresh_supabase_schema():
    """Force refresh Supabase schema cache"""
    try:
        # Make a simple query to refresh schema
        supabase.table('configs').select('id').limit(1).execute()
        logger.info("Supabase schema cache refreshed")
        return True
    except Exception as e:
        logger.error(f"Failed to refresh schema: {e}")
        return False
    

# Load .env file
load_dotenv()

# Flask setup
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '/static/uploads/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'dev-secret-key')

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'} 

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# OpenRouter API Key
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
logger.debug(f"OPENROUTER_API_KEY: {'set' if OPENROUTER_API_KEY else 'not set'}")
if not OPENROUTER_API_KEY:
    logger.error("OPENROUTER_API_KEY not set. Please set it in the .env file or environment.")

# Flask-Limiter setup (using in-memory storage)
if REDIS_AVAILABLE and redis_client:
    limiter = Limiter(
        app=app,
        key_func=get_remote_address,
        default_limits=["100 per day", "10 per minute"],
        storage_uri=f"redis://{os.getenv('REDIS_HOST', 'localhost')}:{os.getenv('REDIS_PORT', 6379)}"
    )
    logger.debug("Flask-Limiter configured with Redis storage.")
else:
    limiter = Limiter(
        app=app,
        key_func=get_remote_address,
        default_limits=["100 per day", "10 per minute"],
        storage_uri="memory://"
)
    logger.warning("Flask-Limiter using in-memory storage (Redis unavailable)")

# ============================================
# Helper Functions
# ============================================

def allowed_file(filename: str) -> bool:
    """Return True if the filename has an allowed extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('access_token'):
            flash('Please log in to access this page.', 'error')
            return redirect(url_for('login'))
        try:
            supabase.auth.get_user(session['access_token'])
        except:
            flash('Session expired. Please log in again.', 'error')
            session.clear()
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def get_current_user():
    """Get current user info from session with robust handling and logging."""
    try:
        logger.debug("get_current_user() session keys: %s", dict(session))
    except Exception:
        logger.debug("get_current_user() could not stringify session")

    access_token = session.get('access_token')
    if not access_token:
        logger.debug("No access_token in session")
        return None

    # Try common supabase client methods to get user info
    try:
        # Preferred: object-like API
        try:
            user_resp = supabase.auth.get_user(access_token)
            logger.debug("supabase.auth.get_user returned (object-like): %s", repr(user_resp))
            # If response is object-like with .user attribute
            user_obj = getattr(user_resp, 'user', None) or (user_resp.get('user') if isinstance(user_resp, dict) else None)
        except Exception as e_inner:
            logger.debug("supabase.auth.get_user(object) raised: %s", e_inner, exc_info=True)
            user_obj = None

        # Fallback: try supabase.auth.api.get_user (older client names)
        if not user_obj:
            try:
                user_resp2 = supabase.auth.get_user(access_token)
                logger.debug("supabase.auth.get_user returned (fallback): %s", repr(user_resp2))
                user_obj = getattr(user_resp2, 'user', None) or (user_resp2.get('user') if isinstance(user_resp2, dict) else None)
            except Exception as e_api:
                logger.debug("supabase.auth.get_user (fallback) raised: %s", e_api, exc_info=True)

        # Final fallback: if supabase returned raw dict in session/token, try to decode token or get profile
        if not user_obj:
            logger.debug("No user object returned by supabase client. Attempting to read profile from Supabase 'profiles' table using session user_id if present.")
            # If you stored user_id in session earlier, try to look up profile
            user_id = session.get('user_id')
            if user_id:
                try:
                    profile_response = supabase.table("profiles").select("*").eq("id", user_id).execute()
                    data = getattr(profile_response, 'data', None) or (profile_response if isinstance(profile_response, list) else None)
                    if data:
                        logger.debug("Profile lookup via session user_id succeeded")
                        profile = data[0] if isinstance(data[0], dict) else {'user_name': str(data[0])}
                        return {
                            'email': session.get('email'),
                            'user_id': user_id,
                            'user_name': profile.get('user_name') if isinstance(profile, dict) else str(profile),
                            'profile': profile
                        }
                except Exception as e_prof:
                    logger.debug("Profile lookup failed: %s", e_prof, exc_info=True)

        # If we have user_obj, try to fetch profile
        if user_obj:
            try:
                uid = getattr(user_obj, 'id', None) or (user_obj.get('id') if isinstance(user_obj, dict) else None)
                profile_response = supabase.table("profiles").select("*").eq("id", uid).execute()
                if profile_response and getattr(profile_response, 'data', None):
                    logger.debug("Profile retrieved from supabase for uid=%s", uid)
                    return {
                        'email': session.get('email'),
                        'user_id': uid,
                        'user_name': profile_response.data[0].get('user_name'),
                        'profile': profile_response.data[0]
                    }
                else:
                    # return minimal user info if profile not present
                    return {
                        'email': session.get('email'),
                        'user_id': uid,
                        'user_name': None,
                        'profile': None
                    }
            except Exception as e_lookup:
                logger.error("Error fetching profile for user: %s", e_lookup, exc_info=True)
                return {
                    'email': session.get('email'),
                    'user_id': uid if 'uid' in locals() else None,
                    'user_name': None,
                    'profile': None
                }

    except Exception as e:
        logger.error("Unexpected error in get_current_user(): %s", e, exc_info=True)
        return None

    logger.debug("get_current_user() returning None after all attempts")
    return None

# ============================================
# Authentication Routes
# ============================================

@app.route('/', methods=['GET', 'POST'])
def login():
    # If already logged in, redirect to AI upload
    if session.get('access_token'):
        return redirect(url_for('ai_upload'))

    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        remember = request.form.get('remember') == 'on'

        logger.info("Login attempt for email: %s", email)

        if not email or not password:
            flash('Email and password are required.', 'error')
            return render_template('login.html')

        try:
            # Call Supabase sign-in and log the raw response for debugging
            response = supabase.auth.sign_in_with_password({"email": email, "password": password})
            logger.debug("Raw sign_in response: %s", repr(response))

            access_token = None
            refresh_token = None
            user_id = None

            # Try object-like attributes (supabase client may return object-like)
            try:
                sess = getattr(response, 'session', None)
                if sess:
                    access_token = getattr(sess, 'access_token', None) or (sess.get('access_token') if isinstance(sess, dict) else None)
                    refresh_token = getattr(sess, 'refresh_token', None) or (sess.get('refresh_token') if isinstance(sess, dict) else None)
                user_obj = getattr(response, 'user', None)
                if user_obj:
                    user_id = getattr(user_obj, 'id', None) or (user_obj.get('id') if isinstance(user_obj, dict) else None)
            except Exception:
                logger.debug("No object-like session/user on response", exc_info=True)

            # Try dict-like response shapes
            if not access_token and isinstance(response, dict):
                session_obj = response.get('session') or response.get('data', {}).get('session')
                user_obj = response.get('user') or response.get('data', {}).get('user')
                if session_obj:
                    access_token = session_obj.get('access_token')
                    refresh_token = session_obj.get('refresh_token')
                if user_obj:
                    user_id = user_obj.get('id')

            # Final fallback checks
            if not access_token:
                try:
                    access_token = response.get('access_token') if isinstance(response, dict) else None
                except Exception:
                    access_token = None

            if not access_token:
                logger.error("Sign-in failed or returned no session token for email=%s. Full response: %s", email, repr(response))
                flash('Login failed: invalid credentials or account not confirmed. Check your email or contact support.', 'error')
                return render_template('login.html')

            # Save tokens in session
            session['access_token'] = access_token
            if refresh_token:
                session['refresh_token'] = refresh_token
            if user_id:
                session['user_id'] = user_id
            session['email'] = email
            session.permanent = remember

            logger.info("Login successful for user_id: %s", user_id or "unknown")
            flash('Login successful!', 'success')
            return redirect(url_for('ai_upload'))

        except Exception as e:
            logger.error("Login error: %s", e, exc_info=True)
            # Extract readable message where possible
            err_text = str(e)
            try:
                if hasattr(e, 'args') and e.args:
                    err_text = e.args[0]
                elif str(e):
                    err_text = str(e)
            except Exception:
                pass
            # Flash the Supabase error for development clarity
            flash(f'Login failed: {err_text}', 'error')
            return render_template('login.html')

    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        user_name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm = request.form.get('confirm')
        terms = request.form.get('terms') == 'on'

        # ✅ Validation checks
        if not all([user_name, email, password, confirm]):
            flash('All fields are required.', 'error')
        elif not terms:
            flash('You must accept the terms and conditions.', 'error')
        elif password != confirm:
            flash('Passwords do not match.', 'error')
        elif len(password) < 6:
            flash('Password must be at least 6 characters long.', 'error')
        else:
            try:
                # ✅ Attempt sign-up
                response = supabase.auth.sign_up({
                    "email": email,
                    "password": password
                })#type: ignore

                user = response.user
                session_data = response.session

                if user:
                    # ✅ Create profile row in database
                    supabase.table("profiles").insert({
                        "id": user.id,
                        "user_name": user_name,
                        "email": email
                    }).execute()

                    logger.info(f"New user created: {user.id}")

                    # ✅ Manage session safely
                    if session_data:
                        session['access_token'] = session_data.access_token
                        session['refresh_token'] = session_data.refresh_token
                        session['email'] = email
                        session['user_id'] = user.id

                        flash('Registration successful! Welcome!', 'success')
                        return redirect(url_for('ai_upload'))
                    else:
                        flash('Registered successfully! Please check your email to verify your account.', 'success')
                        return redirect(url_for('login'))
                else:
                    flash('Registration failed. Please try again.', 'error')

            except Exception as e:
                logger.error(f"Registration error: {e}")
                flash('Registration failed. Email may already exist or service error occurred.', 'error')

    # Render the registration form (GET request)
    return render_template('register.html')

@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form.get('email')
        if not email:
            flash('Please enter an email address.', 'error')
        else:
            try:
                # Add redirect URL for localhost
                redirect_url = request.url_root + 'reset-password'
                supabase.auth.reset_password_for_email(
                    email,
                    options={
                        'redirect_to': redirect_url
                    }
                )
                flash('Password reset email sent! Check your inbox.', 'success')
                session['reset_email'] = email
            except Exception as e:
                flash('Email not found or failed to send.', 'error')
                logger.error(f"Reset error: {e}")

    return render_template('forgot_password.html')

@app.route('/reset-password', methods=['GET', 'POST'])
def reset_password():
    # Get the token from URL parameters (Supabase sends this)
    access_token = request.args.get('access_token') or request.args.get('token')
    
    if access_token and request.method == 'GET':
        # Store the token in session for the POST request
        session['reset_token'] = access_token
        try:
            # Set the session with the token
            supabase.auth.set_session(access_token, access_token)
        except Exception as e:
            logger.error(f"Error setting session: {e}")
            flash('Invalid or expired reset link.', 'error')
            return redirect(url_for('forgot_password'))
    
    if request.method == 'POST':
        new_password = request.form.get('new_password')
        confirm_password = request.form.get('confirm_password')

        if not new_password or not confirm_password:
            flash('Both password fields are required.', 'error')
        elif new_password != confirm_password:
            flash('Passwords do not match.', 'error')
        elif len(new_password) < 6:  # new_password is guaranteed to be str here
            flash('Password must be at least 6 characters.', 'error')
        else:
            try:
                # Get the token from session
                reset_token = session.get('reset_token')
                if not reset_token:
                    flash('No reset token found.', 'error')
                    return redirect(url_for('forgot_password'))

                # Set session again to ensure we're authenticated
                supabase.auth.set_session(reset_token, reset_token)
                
                # Update the password - new_password is guaranteed to be str here
                supabase.auth.update_user({"password": str(new_password)})
                
                # Clear the reset token from session
                session.pop('reset_token', None)
                
                flash('Password reset successfully! Please log in.', 'success')
                return redirect(url_for('login'))
            except Exception as e:
                flash('Invalid or expired reset link.', 'error')
                logger.error(f"Update password error: {e}")

    return render_template('reset_password.html')

@app.route('/dashboard')
@login_required
def dashboard():
    user = get_current_user()
    uploaded_file_url = None
    
    return render_template('dashboard.html', 
                          user=user,
                          uploaded_file=uploaded_file_url)

@app.route('/logout')
def logout():
    try:
        supabase.auth.sign_out()
    except:
        pass
    session.clear()
    flash('Logged out successfully.', 'success')
    return redirect(url_for('login'))

def detect_walls(image_array):
    """More robust wall detection with error handling"""
    try:
        if len(image_array.shape) == 3:
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_array
            
        # Multiple detection methods
        thresh1 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY_INV, 11, 2)
        edges = cv2.Canny(gray, 50, 150)
        
        kernel = np.ones((3,3), np.uint8)
        cleaned_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        combined = cv2.bitwise_or(thresh1, cleaned_edges)
        
        kernel = np.ones((2,2), np.uint8)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
        
        return combined
    except Exception as e:
        logger.error(f"Wall detection failed: {e}")
        # Return a simple fallback
        return np.zeros(image_array.shape[:2], dtype=np.uint8)


def calculate_room_centers(rooms):
    centers = []
    for room_mask in rooms:
        y, x = np.where(room_mask == 255)
        if len(x) > 0 and len(y) > 0:
            center_x = int(np.mean(x))
            center_y = int(np.mean(y))
            centers.append((center_x, center_y))
    return centers

def detect_rooms(image_array, wall_mask, min_area_ratio=0.01):
    """Detect rooms in the blueprint with wall mask"""
    try:
        h, w = image_array.shape[:2]
        min_area = int(h * w * min_area_ratio)
        
        # Create a mask for flood filling
        room_mask = wall_mask.copy() if wall_mask is not None else np.zeros((h, w), dtype=np.uint8)
        
        # Flood fill from corners to identify exterior
        if room_mask.size > 0:
            cv2.floodFill(room_mask, None, (0, 0), 255)
            cv2.floodFill(room_mask, None, (w-1, h-1), 255)
            cv2.floodFill(room_mask, None, (0, h-1), 255)
            cv2.floodFill(room_mask, None, (w-1, 0), 255)
        
        interior = cv2.bitwise_not(room_mask)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(interior, connectivity=8)
        
        rooms = []
        room_centroids = []
        
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                room_mask = np.zeros_like(interior)
                room_mask[labels == i] = 255
                rooms.append(room_mask)
                room_centroids.append(centroids[i])
        
        return rooms, room_centroids
    except Exception as e:
        logger.error(f"Room detection failed: {e}")
        return [], []

# ============================================
# AI Router Placement System
# ============================================

class AIRouterPlacer:
    def __init__(self):
        self.max_retries = 3
        
    def analyze_blueprint_and_place_routers(self, image_path, budget, image_width, image_height):
        """Main method to analyze blueprint and place routers using AI"""
        logger.info(f"Starting AI analysis for blueprint: {image_path}, budget: {budget} ZAR")
        
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        prompt = self._create_router_placement_prompt(image_width, image_height, budget)
        
        for attempt in range(self.max_retries):
            try:
                ai_response = self._query_ai_for_placement(prompt, base64_image)
                
                # Normalize ai_response so we can safely call string methods on it
                try:
                    if isinstance(ai_response, bytes):
                        ai_response = ai_response.decode('utf-8', errors='ignore')
                    elif not isinstance(ai_response, str):
                        # Try to serialize dicts/lists to a JSON string; fallback to str()
                        try:
                            ai_response = json.dumps(ai_response)
                        except Exception:
                            ai_response = str(ai_response)
                except Exception:
                    # As a last resort coerce to string
                    try:
                        ai_response = str(ai_response)
                    except Exception:
                        ai_response = ""

                if ai_response and not ai_response.startswith("[OpenRouter Error]"):
                    router_data = self._parse_ai_response(ai_response, image_width, image_height)
                    
                    if router_data and router_data.get('routers'):
                        logger.info(f"Successfully parsed {len(router_data['routers'])} routers from AI response")
                        return router_data
                    else:
                        logger.warning(f"AI response parsing failed on attempt {attempt + 1}")
                else:
                    logger.warning(f"AI query failed on attempt {attempt + 1}: {ai_response}")
                    
            except Exception as e:
                logger.error(f"Error in AI analysis attempt {attempt + 1}: {e}")
                
            if attempt < self.max_retries - 1:
                time.sleep(2)
        
        logger.warning("AI placement failed, using fallback placement")
        return self._fallback_placement(image_path, budget, image_width, image_height)
    
    def _create_router_placement_prompt(self, width, height, budget):
        return f"""
        You are a network architecture expert. Analyze this blueprint image and provide optimal router placement.

        IMAGE DETAILS:
        - Dimensions: {width}x{height} pixels
        - Budget: {budget} ZAR (South African Rands)
        - Router cost: ~1000 ZAR each

        TASK:
        1. Analyze the blueprint layout, room sizes, and wall positions
        2. Determine optimal number of routers (1-10) within budget
        3. Place routers to maximize coverage while minimizing cost
        4. Avoid placing routers on walls or in inaccessible areas
        5. Prioritize central positions in large rooms and high-traffic areas

        OUTPUT FORMAT (JSON only):
        {{
            "number_of_routers": <integer between 1-10>,
            "total_cost": <calculated total cost in ZAR>,
            "routers": [
                {{"x": <x-coordinate>, "y": <y-coordinate>, "room": "description"}},
                ...
            ],
            "coverage_strategy": "brief explanation of placement strategy",
            "reasoning": "detailed technical explanation"
        }}

        COORDINATE RANGE:
        - x: 0 to {width-1}
        - y: 0 to {height-1}

        IMPORTANT: 
        - Return ONLY valid JSON, no additional text
        - Ensure coordinates are within image bounds
        - Place routers in open areas, not on walls
        - Consider signal propagation through walls
        """
    
    def _query_ai_for_placement(self, prompt, base64_image):

        if not OPENROUTER_API_KEY:
            logger.error("OPENROUTER_API_KEY is not configured")
            return "[OpenRouter Error]: API key not configured"
        
        logger.info(f"OpenRouter API Key: {'*' * 10 + OPENROUTER_API_KEY[-4:] if OPENROUTER_API_KEY else 'NOT SET'}")

        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        }
        vision_models = [
            "google/gemini-2.0-flash-exp:free",
            "google/gemini-flash-1.5:free", 
            "qwen/qwen-2.5-vl-72b-instruct:free",
            "microsoft/visualgen-7b-free"
        ]
        
        # Try each vision model in order until one succeeds
        for model in vision_models:
            try:
                payload = {
                    "model": model,
                    "messages": [
                        {
                            "role": "system", 
                            "content": "You are a professional network architect. Provide only valid JSON responses for router placement analysis."
                        },
                        {
                            "role": "user", 
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}"
                                    }
                                }
                            ]
                        }
                    ],
                    "temperature": 0.1,
                    "max_tokens": 2000
                }

                response = requests.post(url, headers=headers, json=payload, timeout=4500)
                if response.status_code == 200:
                    result = response.json()
                    return result["choices"][0]["message"]["content"].strip()
                else:
                    logger.error(f"OpenRouter API error for model {model}: {response.status_code} - {response.text}")
                    # try the next model in the list
                    continue

            except Exception as e:
                logger.error(f"OpenRouter request failed for model {model}: {e}", exc_info=True)
                # try the next model
                continue

        # If we get here, all models failed
        logger.error("All vision models failed to return a valid response")
        return "[OpenRouter Error]: All models failed"
    
    def _parse_ai_response(self, ai_response, image_width, image_height):
        try:
            cleaned_response = self._clean_ai_response(ai_response)
            router_data = json.loads(cleaned_response)
            
            if not isinstance(router_data, dict):
                raise ValueError("AI response is not a JSON object")
            
            routers = router_data.get('routers', [])
            if not isinstance(routers, list):
                raise ValueError("Routers field is not a list")
            
            validated_routers = []
            for router in routers:
                if isinstance(router, dict) and 'x' in router and 'y' in router:
                    x = int(router['x'])
                    y = int(router['y'])
                    x = max(0, min(x, image_width - 1))
                    y = max(0, min(y, image_height - 1))
                    validated_routers.append((x, y))
            
            if not validated_routers:
                raise ValueError("No valid routers found in AI response")
            
            router_data['routers'] = validated_routers
            router_count = min(router_data.get('number_of_routers', len(validated_routers)), 10)
            
            if len(validated_routers) > router_count:
                router_data['routers'] = validated_routers[:router_count]
            elif len(validated_routers) < router_count:
                router_data['number_of_routers'] = len(validated_routers)
            
            return router_data
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
            return self._extract_routers_from_text(ai_response, image_width, image_height)
        except Exception as e:
            logger.error(f"Error parsing AI response: {e}")
            return None
    
    def _clean_ai_response(self, response):
        if not response:
            return response
        
        cleaned = re.sub(r'```json\s*', '', response)
        cleaned = re.sub(r'```\s*', '', cleaned)
        cleaned = cleaned.strip()
        cleaned = cleaned.replace("'", '"')
        cleaned = re.sub(r',\s*}', '}', cleaned)
        cleaned = re.sub(r',\s*]', ']', cleaned)
        
        json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
        if json_match:
            cleaned = json_match.group(0)
        
        return cleaned
    
    def _extract_routers_from_text(self, text, image_width, image_height):
        try:
            routers = []
            pattern1 = r'["\']?x["\']?\s*:\s*(\d+)\s*,\s*["\']?y["\']?\s*:\s*(\d+)'
            matches1 = re.findall(pattern1, text, re.IGNORECASE)
            for x, y in matches1:
                routers.append((int(x), int(y)))
            
            unique_routers = []
            for x, y in routers:
                x = max(0, min(int(x), image_width - 1))
                y = max(0, min(int(y), image_height - 1))
                if (x, y) not in unique_routers:
                    unique_routers.append((x, y))
            
            if unique_routers:
                return {
                    'number_of_routers': len(unique_routers),
                    'routers': unique_routers,
                    'coverage_strategy': 'Extracted from text analysis',
                    'reasoning': 'Router positions extracted using pattern matching'
                }
        except Exception as e:
            logger.error(f"Error extracting routers from text: {e}")
        
        return None    

    
    def _fallback_placement(self, image_path, budget, image_width, image_height):
        """Robust fallback placement with proper error handling"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Failed to load image for fallback placement")
                
            image_array = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            wall_mask = detect_walls(image_array)
            rooms, room_centroids = detect_rooms(image_array, wall_mask)
            
            # Use room centroids or fallback to grid placement
            if room_centroids and len(room_centroids) > 0:
                room_centers = [(int(cent[0]), int(cent[1])) for cent in room_centroids]
            else:
                # Grid-based fallback
                room_centers = []
                grid_x = max(2, image_width // 200)
                grid_y = max(2, image_height // 200)
                for i in range(1, grid_x):
                    for j in range(1, grid_y):
                        x = (i * image_width) // grid_x
                        y = (j * image_height) // grid_y
                        room_centers.append((x, y))
            
            if not room_centers:
                room_centers = [(image_width // 2, image_height // 2)]
            
            router_cost = 1000
            max_routers = min(len(room_centers), budget // router_cost, 10)
            max_routers = max(1, max_routers)
            routers = room_centers[:max_routers]
            
            return {
                'number_of_routers': len(routers),
                'routers': routers,
                'coverage_strategy': 'Fallback: Centered in detected rooms',
                'reasoning': 'AI analysis failed, using room-based fallback placement'
            }
        except Exception as e:
            logger.error(f"Fallback placement failed: {e}")
            return {
                'number_of_routers': 1,
                'routers': [(image_width // 2, image_height // 2)],
                'coverage_strategy': 'Ultimate fallback: Center placement',
                'reasoning': 'All placement methods failed, using center of image'
            }

ai_router_placer = AIRouterPlacer()

def preprocess_image_for_ai(image_path):
    """Resize and optimize image for AI analysis"""
    with Image.open(image_path) as img:
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize if too large (keep aspect ratio)
        max_size = 1024
        if max(img.size) > max_size:
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        # Save optimized version
        temp_path = image_path + "_optimized.jpg"
        img.save(temp_path, "JPEG", quality=85, optimize=True)
        return temp_path

def query_openrouter(
    prompt: str,
    base64_image: str | None = None,
    for_image_analysis: bool = False,
) -> str:
    """
    Call OpenRouter with a **1200-second timeout** (20 min) for text,
    **1500 seconds** for vision models.
    """
    if not OPENROUTER_API_KEY:
        return "[OpenRouter Error]: API key missing"

    # ---- model priority ----------------------------------------------------
    if for_image_analysis:
        models = [
            "google/gemini-2.0-flash-exp:free",
            "qwen/qwen-2.5-vl-72b-instruct:free",
            "google/gemini-flash-1.5:free",
        ]
    else:
        models = [
            
            "google/gemini-flash-1.5:free",
            "meta-llama/llama-3.1-8b-instruct:free",
            "google/gemini-2.0-flash-exp:free",
            "qwen/qwen-2.5-vl-72b-instruct:free",
        ]

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:5000",
        "X-Title": "AI Network Planner",
    }

    # ---- build message payload --------------------------------------------
    message_content: list[dict] = [{"type": "text", "text": prompt}]
    if base64_image and for_image_analysis:
        # Default to jpeg unless you have a way to detect PNG
        mime_type = 'image/jpeg'
        # If you want to detect PNG, you could add a parameter or check the image header
        message_content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{base64_image}"},
            }
        )
        
    
    for model in models[:3]:
        timeout = 1500 if for_image_analysis else 1200   # <-- **HERE** is the longer time
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a professional AI network architect. "
                        "Give concise, practical advice in South African Rands (ZAR)."
                    ),
                },
                {"role": "user", "content": message_content},
            ],
            "max_tokens": 12000,
            "temperature": 0.3,
        }

        for attempt in range(1, 4):   # max 3 retries per model
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
                if resp.status_code == 429:                     # rate-limit
                    delay = float(resp.headers.get("Retry-After", 5 * attempt))
                    logger.warning(f"Rate-limited ({model}), waiting {delay}s")
                    time.sleep(delay)
                    continue

                if resp.status_code == 200:
                    reply = resp.json()["choices"][0]["message"]["content"].strip()
                    # cache the result (if you have a cache helper)
                    cache_key = hashlib.md5(prompt.encode()).hexdigest()
                    # store_chat_cache(cache_key, reply)   # uncomment if you have it
                    logger.info(f"OpenRouter success with {model}")
                    return reply

                logger.warning(f"OpenRouter {model} error {resp.status_code}: {resp.text}")
                break   # non-retryable

            except requests.Timeout:
                logger.warning(f"Timeout ({timeout}s) for {model}, attempt {attempt}")
            except requests.RequestException as e:
                logger.warning(f"Request exception for {model}: {e}")

            if attempt < 3:
                time.sleep(2 ** attempt)   # exponential back-off

    # ---- all attempts failed ------------------------------------------------
    return get_simple_fallback_response(prompt)

# ============================================
# AI Routes
# ============================================

@app.route('/ai/', methods=['GET', 'POST'])
@login_required
def ai_upload():
    user = get_current_user()
    
    if not user:
        logger.warning("get_current_user() returned None despite login_required. Redirecting to login.")
        session.clear()
        flash('Session invalid. Please log in again.', 'error')
        return redirect(url_for('login'))

    # Safe visitor logging - FIXED VERSION
    def safe_log_visitor(ip_address: str):
        """Safely log visitor with error handling"""
        try:
            success = log_visitor(ip_address)
            if not success:
                logger.warning(f"Visitor logging failed for: {ip_address}")
        except Exception as e:
            logger.warning(f"Visitor logging error (non-critical): {e}")

    safe_log_visitor(request.remote_addr if request.remote_addr is not None else "unknown")

    if request.method == 'POST':
        try:
            logger.info("=== Upload Request Started ===")
            logger.debug("Request form keys: %s", list(request.form.keys()))
            logger.debug("Request files keys: %s", list(request.files.keys()))

            # Validate presence of file
            if 'image' not in request.files:
                logger.error("No image file in request.files")
                flash('No image file provided', 'error')
                return render_template('index.html', user=user)

            file = request.files['image']
            logger.info("Uploaded filename: %s", getattr(file, 'filename', None))

            if not file or file.filename == '':
                logger.error("Empty filename or no file provided")
                flash('No file selected', 'error')
                return render_template('index.html', user=user)

            # Ensure filename is a string before passing to allowed_file to satisfy type checkers
            if not allowed_file(file.filename or ""):
                logger.error("Disallowed file extension: %s", getattr(file, 'filename', 'None'))
                flash('Invalid file type. Please upload a PNG or JPEG image.', 'error')
                return render_template('index.html', user=user)

            # Validate budget
            try:
                budget = request.form.get('budget', type=int)
            except Exception:
                budget = None
            logger.info("Budget provided: %s", budget)

            if not budget or budget < 2500:
                logger.error("Invalid or missing budget: %s", budget)
                flash('Budget must be at least 2,500 ZAR', 'error')
                return render_template('index.html', user=user)

            # Save file safely
            filename = secure_filename(file.filename if file.filename is not None else "uploaded_image.png")
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            try:
                file.save(file_path)
                
                # Load and validate the saved image
                image = cv2.imread(file_path, cv2.IMREAD_COLOR)
                if image is None:
                    logger.error(f"Failed to load image: {file_path}")
                    return render_template('result.html', error='Failed to load image.')
                image_array = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            except Exception as e:
                logger.error("Error saving or loading file: %s", e, exc_info=True)
                flash('Server error processing file', 'error')
                return render_template('index.html', user=user)


            if os.path.exists(file_path):
                logger.info(f"File successfully saved to: {file_path}")
            else:
                logger.error(f"File save failed - path does not exist: {file_path}")
                raise FileNotFoundError(f"Failed to save file at: {file_path}")

            # Load image with cv2 and validate
            image = cv2.imread(file_path, cv2.IMREAD_COLOR)
            if image is None:
                logger.error("cv2.imread returned None for %s", file_path)
                flash('Failed to load image. Ensure it\'s a valid PNG or JPEG.', 'error')
                # remove invalid uploaded file to avoid reusing later
                try:
                    os.remove(file_path)
                except Exception:
                    pass
                return render_template('index.html', user=user)

            image_array = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w = image_array.shape[:2]
            logger.info("Image dimensions: %dx%d", w, h)

            # Run AI analysis with robust guarding
            try:
                router_data = ai_router_placer.analyze_blueprint_and_place_routers(file_path, budget, w, h)
                logger.info("AI analysis returned: %s", bool(router_data))
            except Exception as e:
                logger.error("AI analysis raised exception: %s", e, exc_info=True)
                router_data = None

            # Ensure router_data is valid; fallback if necessary
            if not router_data or not isinstance(router_data.get('routers'), (list, tuple)) or len(router_data.get('routers', [])) == 0:
                logger.warning("Invalid router_data received from AI; using fallback placement")
                try:
                    router_data = ai_router_placer._fallback_placement(file_path, budget, w, h)
                except Exception as e:
                    logger.error("Fallback placement failed: %s", e, exc_info=True)
                    # Last-ditch fallback
                    router_data = {
                        'number_of_routers': 1,
                        'routers': [(w // 2, h // 2)],
                        'coverage_strategy': 'ultimate fallback',
                        'reasoning': 'forced fallback due to errors'
                    }

            # Generate config key (guard access to user id)
            try:
                uid = user.get('user_id') or session.get('user_id') or 'anonymous'
            except Exception:
                uid = session.get('user_id') or 'anonymous'
            config_key = hashlib.md5(f"{filename}_{budget}_{uid}".encode()).hexdigest()
            logger.info("Config key: %s", config_key)

            # Image processing: detect walls, simulate and overlay
            try:
                wall_mask = detect_walls(image_array)
                if wall_mask.shape[:2] != (h, w):
                    wall_mask = cv2.resize(wall_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            except Exception as e:
                logger.error("Wall detection failed: %s", e, exc_info=True)
                wall_mask = np.zeros((h, w), dtype=np.uint8)

            coverage_img_filename = f"coverage_{config_key}.png"
            router_img_filename   = f"router_{config_key}.png"
            coverage_img_path = os.path.join(app.config['UPLOAD_FOLDER'], coverage_img_filename)
            router_img_path   = os.path.join(app.config['UPLOAD_FOLDER'], router_img_filename)

            # ---- DEFAULT TO PLACEHOLDER (will be overwritten if save succeeds) ----
            coverage_img_filename = router_img_filename = "placeholder.png"

            try:
                # 1. Simulate the signal
                coverage_map = simulate_signal(image_array, router_data['routers'], wall_mask)
                coverage_map = np.clip(coverage_map.astype(np.float32), 0, 1)

                # 2. Save the two images
                overlay_signal(image_array, coverage_map, router_data['routers'],
                            coverage_img_path, router_img_path)

                # 3. Verify the files really exist
                if not (os.path.isfile(coverage_img_path) and os.path.isfile(router_img_path)):
                    raise FileNotFoundError("One of the generated images is missing")

                # 4. If we get here → everything is fine → use the real filenames
                coverage_img_filename = f"coverage_{config_key}.png"
                router_img_filename   = f"router_{config_key}.png"
                logger.info(f"Images saved: {coverage_img_path}, {router_img_path}")

            except Exception as e:
                logger.error(f"Image generation failed – using placeholder: {e}", exc_info=True)
                # keep the placeholder filenames that were set at the top of the block

                try:
                    create_placeholder_image()
                    coverage_img_filename = 'placeholder.png'
                    router_img_filename = 'placeholder.png'
                except Exception:
                    coverage_img_filename = 'placeholder.png'
                    router_img_filename = 'placeholder.png'

            # Store configuration (non-blocking)
            try:
                store_config(config_key, {
                    'routers': router_data['routers'],
                    'budget': budget,
                    'filename': filename,
                    'width': w,
                    'height': h,
                    'router_data': router_data,
                    'user_id': uid
                })
            except Exception as e:
                logger.error("store_config failed: %s", e, exc_info=True)

            logger.info("=== Upload Request Completed Successfully ===")
            return render_template(
                'result.html',
                router_img=router_img_filename,
                coverage_img=coverage_img_filename,
                config_key=config_key,
                blueprint_filename=filename,
                ai_analysis=router_data,
                user=user
            )

        except Exception as e:
            logger.error("=== CRITICAL ERROR in ai_upload ===: %s", e, exc_info=True)
            flash(f'Server error: {str(e)}', 'error')
            return render_template('index.html', user=user)
        

    return render_template('index.html', user=user)



@app.route('/ai/stream-suggestion')
@limiter.limit("5 per minute")
def stream_suggestion():
    """Stream AI suggestions via Server-Sent Events with better error handling"""
    logger.debug("stream_suggestion called; request.args: %s", dict(request.args))
    
    def error_stream(error_msg):
        logger.debug("stream_suggestion error: %s", error_msg)
        yield f"data: {json.dumps({'error': error_msg, 'done': True})}\n\n"

    config_key = request.args.get('config_key', '').strip()
    logger.debug("stream_suggestion config_key: '%s'", config_key)

    if not config_key:
        return Response(error_stream('No config_key provided'), mimetype='text/event-stream')

    # Try to get cached suggestion first
    cached_response = get_suggestion(config_key)
    logger.debug("stream_suggestion cached_response: %s", bool(cached_response))

    if cached_response:
        def cached_stream():
            chunk_size = 20
            for i in range(0, len(cached_response), chunk_size):
                chunk = cached_response[i:i + chunk_size]
                yield f"data: {json.dumps({'chunk': chunk, 'done': False})}\n\n"
                time.sleep(0.01)
            yield f"data: {json.dumps({'done': True})}\n\n"
        return Response(cached_stream(), mimetype='text/event-stream')

    # Get configuration
    config = get_config(config_key)
    if not config:
        logger.error(f"No configuration found for key: {config_key}")
        return Response(error_stream('No configuration available'), mimetype='text/event-stream')

    router_data = config.get('router_data', {})
    logger.debug("Router data found: %s", bool(router_data))
    
    def stream_ai_analysis():
        try:
            full_response = ""
            
            if router_data:
                analysis_text = f"""AI Network Analysis Summary:

            Number of Routers: {router_data.get('number_of_routers', 'N/A')}
            Total Cost: {router_data.get('total_cost', 'N/A')} ZAR
            Placement Strategy: {router_data.get('coverage_strategy', 'N/A')}

            Technical Reasoning:
            {router_data.get('reasoning', 'No detailed reasoning provided.')}

            Router Positions:
            {router_data.get('routers', [])}

            Recommendations:
            - Ensure routers are placed in accessible locations
            - Consider cable routing and power outlets
            - Test signal strength in all rooms after installation
            - Consider future expansion needs
            """
                full_response = analysis_text
            else:
                full_response = "AI analysis data not available. Please try uploading the blueprint again."
            
            # Stream the response
            chunk_size = 20
            for i in range(0, len(full_response), chunk_size):
                chunk = full_response[i:i + chunk_size]
                yield f"data: {json.dumps({'chunk': chunk, 'done': False})}\n\n"
                time.sleep(0.02)
            
            # Cache the response
            if full_response.strip():
                store_suggestion(config_key, full_response)
                logger.info(f"Cached suggestion for config_key: {config_key}")
            
            yield f"data: {json.dumps({'done': True})}\n\n"
            
        except Exception as e:
            logger.error(f"Error in suggestion streaming: {e}", exc_info=True)
            yield f"data: {json.dumps({'error': 'Failed to generate suggestions', 'done': True})}\n\n"
    
    return Response(stream_ai_analysis(), mimetype='text/event-stream')
# Image processing functions


def simulate_signal(image_array, routers, wall_mask, signal_radius=100):
    """More robust signal simulation"""
    try:
        h, w = image_array.shape[:2]
        coverage_map = np.zeros((h, w), dtype=np.float32)
        
        if not routers:
            logger.warning("No routers provided for signal simulation")
            return coverage_map
            
        for x, y in routers:
            # Ensure coordinates are within bounds
            x = max(0, min(int(x), w-1))
            y = max(0, min(int(y), h-1))
            
            i, j = np.ogrid[max(0, y - signal_radius):min(h, y + signal_radius), 
                            max(0, x - signal_radius):min(w, x + signal_radius)]
            distances = np.sqrt((i - y)**2 + (j - x)**2)
            within_radius = distances <= signal_radius
            
            if not np.any(within_radius):
                continue
            
            signal_strength = np.zeros_like(distances, dtype=np.float32)
            signal_strength[within_radius] = 1.0 - distances[within_radius] / signal_radius
            
            coverage_map[
                max(0, y - signal_radius):min(h, y + signal_radius),
                max(0, x - signal_radius):min(w, x + signal_radius)
            ] = np.maximum(
                coverage_map[
                    max(0, y - signal_radius):min(h, y + signal_radius),
                    max(0, x - signal_radius):min(w, x + signal_radius)
                ],
                signal_strength
            )
        
        return coverage_map
    except Exception as e:
        logger.error(f"Signal simulation failed: {e}")
        return np.zeros(image_array.shape[:2], dtype=np.float32)

def overlay_signal(image_array, coverage_map, routers, coverage_path, router_path, downscale=0.25):
    """More robust image overlay with better error handling"""
    try:
        logger.info(f"Starting image overlay - coverage_path: {coverage_path}, router_path: {router_path}")
        
        # Ensure coverage_map is valid
        if coverage_map is None or coverage_map.size == 0:
            logger.warning("Empty coverage map, creating fallback")
            coverage_map = np.zeros(image_array.shape[:2], dtype=np.float32)
        
        # Create the uploads directory if it doesn't exist
        os.makedirs(os.path.dirname(coverage_path), exist_ok=True)
        logger.info(f"Directory ensured for: {os.path.dirname(coverage_path)}")

        heat = np.array(np.clip(coverage_map, 0, 1) * 255, dtype=np.uint8)
        
        if downscale < 1:
            target_size = (int(heat.shape[1] * downscale), int(heat.shape[0] * downscale))
            small_heat = cv2.resize(heat, target_size, interpolation=cv2.INTER_LINEAR)
            heatmap_small = cv2.applyColorMap(np.ascontiguousarray(small_heat), cv2.COLORMAP_JET)
            heatmap = cv2.resize(heatmap_small, (image_array.shape[1], image_array.shape[0]), interpolation=cv2.INTER_LINEAR)
        else:
            heatmap = cv2.applyColorMap(np.ascontiguousarray(heat), cv2.COLORMAP_JET)
        
        blended = cv2.addWeighted(image_array, 0.7, heatmap, 0.6, 0)
        blended_rgb = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)
        
        # Save with error handling
        try:
            Image.fromarray(blended_rgb).save(coverage_path)
            logger.info(f"Successfully saved coverage image to: {coverage_path}")
            logger.info(f"File exists: {os.path.exists(coverage_path)}")
            logger.info(f"File size: {os.path.getsize(coverage_path) if os.path.exists(coverage_path) else 'N/A'} bytes")
        except Exception as save_error:
            logger.error(f"Failed to save coverage image: {save_error}")
            raise
        
        # Create router placement image
        router_img = image_array.copy()
        for x, y in routers:
            x = int(x)
            y = int(y)
            if 0 <= x < router_img.shape[1] and 0 <= y < router_img.shape[0]:
                cv2.circle(router_img, (x, y), 10, (0, 255, 0), -1)
        
        router_rgb = cv2.cvtColor(router_img, cv2.COLOR_BGR2RGB)
        
        try:
            Image.fromarray(router_rgb).save(router_path)
            logger.info(f"Successfully saved router image to: {router_path}")
            logger.info(f"File exists: {os.path.exists(router_path)}")
            logger.info(f"File size: {os.path.getsize(router_path) if os.path.exists(router_path) else 'N/A'} bytes")
        except Exception as save_error:
            logger.error(f"Failed to save router image: {save_error}")
            raise
            
    except Exception as e:
        logger.error(f"Error in overlay_signal: {e}", exc_info=True)
        # Create fallback images
        try:
            fallback_img = np.zeros((400, 400, 3), dtype=np.uint8)
            Image.fromarray(fallback_img).save(coverage_path)
            Image.fromarray(fallback_img).save(router_path)
            logger.info("Created fallback placeholder images")
        except:
            logger.error("Failed to create fallback images")
        raise





def blueprint_to_3d(image):
    try:
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        cleaned = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel, iterations=2)
        _, binary = cv2.threshold(cleaned, 200, 255, cv2.THRESH_BINARY_INV)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        min_area = 500
        filtered = np.zeros_like(binary)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                filtered[labels == i] = 255
        contours, _ = cv2.findContours(filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        wall_height = 40
        ax.set_box_aspect(tuple(map(float, [image_cv.shape[1], image_cv.shape[0], wall_height]))) 
        ax.set_axis_off()
        for cnt in contours:
            cnt = cnt.squeeze()
            if cnt.ndim != 2 or cnt.shape[0] < 3:
                continue
            base = np.hstack([cnt, np.zeros((cnt.shape[0], 1))])
            top = np.hstack([cnt, np.full((cnt.shape[0], 1), wall_height)])
            faces = []
            for i in range(len(base)):
                next_i = (i + 1) % len(base)
                faces.append([base[i], base[next_i], top[next_i], top[i]])
            faces.append(top)
            poly = Poly3DCollection(faces, facecolors='lightgray', edgecolors='black', linewidths=0.3, alpha=0.95)
            ax.add_collection3d(poly)
        ax.set_xlim(0, image_cv.shape[1])
        ax.set_ylim(image_cv.shape[0], 0)
        ax.set_zlim(0, wall_height)
        ax.view_init(elev=45, azim=45)
        buf = BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        return Image.open(buf)
    except Exception as e:
        logger.error(f"3D rendering error: {e}")
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        return Image.fromarray(blank)


@app.route('/ai/render-3d', methods=['POST'])
@login_required
def render_3d():
    """Render 3D view of blueprint with robust error handling"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
            
        filename = data.get('filename')
        if not filename:
            return jsonify({'error': 'No filename provided'}), 400
            
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(filename))
        if not os.path.exists(filepath):
            return jsonify({'error': 'Blueprint file not found'}), 404
        
        try:
            image = Image.open(filepath)
            render3d_image = blueprint_to_3d(image)
            preview3d_path = os.path.join(app.config['UPLOAD_FOLDER'], 'preview3d.png')
            render3d_image.save(preview3d_path)
            
            return jsonify({'preview_img': 'preview3d.png'})
        except Exception as render_error:
            logger.error(f"3D rendering failed: {render_error}")
            return jsonify({'error': f'3D rendering failed: {str(render_error)}'}), 500
            
    except Exception as e:
        logger.error(f"Error in render_3d route: {e}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500
    

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    try:
        logger.info(f"Attempting to serve file: {filename}")
        
        # Security check
        if '..' in filename or filename.startswith('/'):
            logger.warning(f"Security violation attempt: {filename}")
            return "Invalid filename", 400
            
        uploads_dir = app.config['UPLOAD_FOLDER']
        file_path = os.path.join(uploads_dir, filename)
        
        # Enhanced logging
        logger.info(f"Looking for file: {file_path}")
        logger.info(f"Uploads directory exists: {os.path.exists(uploads_dir)}")
        
        if not os.path.exists(uploads_dir):
            logger.error(f"Uploads directory does not exist: {uploads_dir}")
            return "Uploads directory not found", 500
            
        if not os.path.isfile(file_path):
            logger.warning(f"File not found: {file_path}")
            logger.warning(f"Files in uploads directory: {os.listdir(uploads_dir)}")
            return "File not found", 404
            
        logger.info(f"File found, serving: {file_path}")
        
        # Use send_from_directory instead of send_file for better security
        return send_from_directory(uploads_dir, filename)
        
    except Exception as e:
        logger.error(f"Error serving file {filename}: {str(e)}", exc_info=True)
        return f"Error serving file: {str(e)}", 500
    
@app.route('/ai/download-pdf')
@login_required
def download_pdf():
    """Generate and download PDF report with improved error handling"""
    try:
        config_key = request.args.get('config_key', '').strip()
        if not config_key:
            logger.error("No config_key provided for PDF download")
            flash('No configuration key provided for PDF download', 'error')
            return redirect(request.referrer or url_for('ai_upload'))

        # Get configuration data
        config = get_config(config_key)
        if not config:
            logger.error(f"No configuration found for key: {config_key}")
            flash('No configuration data found for PDF generation', 'error')
            return redirect(request.referrer or url_for('ai_upload'))

        # Get AI suggestion
        suggestion = get_suggestion(config_key)
        if not suggestion:
            logger.warning(f"No suggestion found for key: {config_key}")
            # Use basic analysis data if no suggestion cached
            router_data = config.get('router_data', {})
            suggestion = f"""
                AI Network Analysis Summary:

                Number of Routers: {router_data.get('number_of_routers', 'N/A')}
                Total Cost: {router_data.get('total_cost', 'N/A')} ZAR
                Placement Strategy: {router_data.get('coverage_strategy', 'N/A')}

                Technical Reasoning:
                {router_data.get('reasoning', 'No detailed reasoning provided.')}

                Router Positions:
                {router_data.get('routers', [])}
                """

        buffer = BytesIO()
        c = pdf_canvas.Canvas(buffer, pagesize=A4)
        width, height = A4
        
        # Set up styles
        c.setTitle("Network Architecture Report")
        
        # Header
        c.setFont("Helvetica-Bold", 16)
        c.drawString(50, height - 50, "Network Architecture Report")
        c.setFont("Helvetica", 10)
        c.drawString(50, height - 70, f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        y_position = height - 100
        
        # AI Analysis Section
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y_position, "AI Analysis Summary:")
        y_position -= 20
        
        c.setFont("Helvetica", 10)
        lines = suggestion.split('\n')
        for line in lines:
            if y_position < 100:  # New page if needed
                c.showPage()
                y_position = height - 50
                c.setFont("Helvetica", 10)
            
            # Clean up the line for PDF
            clean_line = line.strip()
            if clean_line:
                c.drawString(50, y_position, clean_line[:80])  # Limit line length
                y_position -= 15
        
        # Add images if they exist
        y_position -= 20
        
        def add_image_to_pdf(image_path, title):
            nonlocal y_position
            if os.path.exists(image_path):
                if y_position < 250:  # Need new page for image
                    c.showPage()
                    y_position = height - 50
                
                c.setFont("Helvetica-Bold", 12)
                c.drawString(50, y_position, title)
                y_position -= 10
                
                try:
                    img = ImageReader(image_path)
                    c.drawImage(img, 50, y_position - 200, width=500, height=200, preserveAspectRatio=True)
                    y_position -= 220
                except Exception as e:
                    logger.error(f"Error adding image {image_path} to PDF: {e}")
                    c.drawString(50, y_position, f"Image unavailable: {str(e)}")
                    y_position -= 20
        
        # Add router placement image
        router_img_path = os.path.join(app.config['UPLOAD_FOLDER'], f"router_{config_key}.png")
        add_image_to_pdf(router_img_path, "Router Placement")
        
        # Add signal heatmap image
        coverage_img_path = os.path.join(app.config['UPLOAD_FOLDER'], f"coverage_{config_key}.png")
        add_image_to_pdf(coverage_img_path, "Signal Coverage Heatmap")
        
        # Add 3D preview if available
        preview3d_path = os.path.join(app.config['UPLOAD_FOLDER'], 'preview3d.png')
        add_image_to_pdf(preview3d_path, "3D Blueprint Preview")
        
        c.save()
        buffer.seek(0)
        
        # Log PDF generation
        logger.info(f"PDF generated successfully for config_key: {config_key}")
        
        return send_file(
            buffer,
            as_attachment=True,
            download_name=f"network_report_{config_key}.pdf",
            mimetype='application/pdf'
        )
        
    except Exception as e:
        logger.error(f"Error generating PDF: {e}", exc_info=True)
        flash(f'Error generating PDF: {str(e)}', 'error')
        return redirect(request.referrer or url_for('ai_upload'))
    


    
# Chat routes

@app.route('/ai/chat', methods=['POST'])
@limiter.limit("5 per minute")
@login_required
def chat():
    if not request.is_json:
        return jsonify({'error': 'Invalid JSON format'}), 400
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        if not user_message:
            return jsonify({'error': 'Empty message'}), 400

        system_prompt = (
            "You are a highly skilled and specialized AI network architect. "
            "Your job is to help users with network planning: router placement, signal optimization, "
            "network topology, cabling, access points, bandwidth management, and related infrastructure. "
            "Always provide costs and budgets in South African Rands (ZAR). "
            "Politely refuse to answer any question outside of this domain."
        )
        full_prompt = f"{system_prompt}\n\nUser: {user_message}\nAI:"
        
        reply = query_openrouter(full_prompt, for_image_analysis=False)
        if reply.startswith("[OpenRouter Error]") or "unavailable" in reply.lower():
            reply = get_simple_fallback_response(user_message)
        
        return jsonify({'response': reply})
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        return jsonify({'error': f'Error: {str(e)}'}), 500
    

@app.route('/ai/chat-stream', methods = ['GET'])
@limiter.limit("5 per minute")
def chat_stream():
    """Stream AI suggestions via Server-Sent Events with proper auth handling"""
    
    # Manual authentication check for SSE endpoints
    def error_stream(error_msg: str):
        yield f"data: {json.dumps({'error': error_msg, 'done': True})}\n\n"

    # Check if user is authenticated
    if not session.get('access_token'):
        logger.warning("Unauthorized access to chat-stream")
        return Response(error_stream('Unauthorized - please log in'), mimetype='text/event-stream')

    # Verify the token is still valid
    try:
        user_resp = supabase.auth.get_user(session['access_token'])
        if not user_resp or not getattr(user_resp, 'user', None):
            logger.warning("Invalid token in chat-stream")
            session.clear()
            return Response(error_stream('Session expired - please log in again'), mimetype='text/event-stream')
    except Exception as e:
        logger.error(f"Token verification failed in chat-stream: {e}")
        session.clear()
        return Response(error_stream('Authentication error - please log in again'), mimetype='text/event-stream')

    try:
        user_message = request.args.get('message', '')
        message_id = request.args.get('messageId', 'unknown')
        
        if not user_message or not user_message.strip():
            return Response(error_stream('Empty message'), mimetype='text/event-stream')

        user_message = re.sub(r'[^\w\s.,!?]', '', user_message.strip())
        logger.info(f"Chat stream request - messageId: {message_id}")

        system_prompt = (
            "You are a highly skilled AI network architect. "
            "Help users with network planning in South African Rands (ZAR). "
            "Keep responses concise and practical."
        )
        full_prompt = f"{system_prompt}\n\nUser: {user_message}\nAI:"
        cache_key = hashlib.md5(full_prompt.encode('utf-8')).hexdigest()

        cached_response = get_chat_cache(cache_key)
        
        if cached_response:
            def cached_stream():
                chunk_size = 20
                for i in range(0, len(cached_response), chunk_size):
                    chunk = cached_response[i:i + chunk_size]
                    yield f"data: {json.dumps({'chunk': chunk, 'done': False})}\n\n"
                    time.sleep(0.01)
                yield f"data: {json.dumps({'done': True})}\n\n"
            
            return Response(cached_stream(), mimetype='text/event-stream')

        # Get response from OpenRouter
        response = query_openrouter(full_prompt, for_image_analysis=False)
        
        # Check if response is an error
        if response.startswith("[OpenRouter Error]:"):
            logger.error(f"OpenRouter error for message '{user_message}': {response}")
            return Response(error_stream("AI service is temporarily unavailable. Please try again in a moment."), mimetype='text/event-stream')
        
        def stream_response():
            chunk_size = 20
            for i in range(0, len(response), chunk_size):
                chunk = response[i:i + chunk_size]
                yield f"data: {json.dumps({'chunk': chunk, 'done': False})}\n\n"
                time.sleep(0.01)
            yield f"data: {json.dumps({'done': True})}\n\n"

        # Cache successful response
        if response and not response.startswith("[OpenRouter Error]:"):
            store_chat_cache(cache_key, response)

        return Response(stream_response(), mimetype='text/event-stream')
    
    except Exception as e:
        logger.error(f"Chat stream error: {e}", exc_info=True)
        return Response(error_stream(f"Server error: {str(e)}"), mimetype='text/event-stream')
    

@app.route('/ai/clear-cache', methods=['POST'])
@login_required
def clear_cache():
    """Clear cache with options for Redis only or both"""
    try:
        data = request.get_json(silent=True)
        clear_type = data.get('type', 'all') if data else 'all'
        
        if clear_type == 'redis':
            success = clear_redis_only()
            message = 'Redis cache cleared successfully' if success else 'Failed to clear Redis cache'
        else:
            success = clear_all_cache()
            message = 'All caches cleared successfully' if success else 'Failed to clear caches'
        
        if success:
            return jsonify({'message': message, 'success': True})
        else:
            return jsonify({'error': message, 'success': False}), 500
            
    except Exception as e:
        logger.error(f"Clear cache error: {e}", exc_info=True)
        # Always return a proper HTTP response on error to satisfy Flask route expectations
        return jsonify({'error': 'Failed to clear cache', 'details': str(e), 'success': False}), 500

@app.route('/api/cache-stats')
@login_required
def cache_stats():
    """Get cache statistics"""
    stats = {
        'redis_available': REDIS_AVAILABLE,
        'storage_status': get_storage_status()
    }
    
    if REDIS_AVAILABLE and redis_client:
        try:
            # Call redis methods and handle both synchronous and asynchronous clients.
            try:
                info_candidate = redis_client.info()
            except TypeError:
                # Some clients expose the method as an attribute that must be called later
                info_candidate = redis_client.info

            # If it's callable, call it (it may return a value or an awaitable)
            if callable(info_candidate):
                try:
                    info_candidate = info_candidate()
                except Exception as e_call_info:
                    logger.debug(f"Calling redis.info() raised: {e_call_info}", exc_info=True)

            # If it's an awaitable, run it in a dedicated event loop and otherwise use the value directly.
            if inspect.isawaitable(info_candidate):
                loop = asyncio.new_event_loop()
                try:
                    info = loop.run_until_complete(info_candidate)
                finally:
                    loop.close()
            else:
                info = info_candidate

            try:
                dbsize_candidate = redis_client.dbsize()
            except TypeError:
                dbsize_candidate = redis_client.dbsize

            if callable(dbsize_candidate):
                try:
                    dbsize_candidate = dbsize_candidate()
                except Exception as e_call_db:
                    logger.debug(f"Calling redis.dbsize() raised: {e_call_db}", exc_info=True)

            if inspect.isawaitable(dbsize_candidate):
                loop = asyncio.new_event_loop()
                try:
                    dbsize = loop.run_until_complete(dbsize_candidate)
                finally:
                    loop.close()
            else:
                dbsize = dbsize_candidate

            # Normalize dbsize to an int when possible
            dbsize_val = dbsize
            try:
                if isinstance(dbsize, (int, float)):
                    dbsize_val = int(dbsize)
                elif isinstance(dbsize, bytes):
                    try:
                        dbsize_val = int(dbsize.decode())
                    except Exception:
                        try:
                            dbsize_val = int(dbsize.decode(errors='ignore'))
                        except Exception:
                            dbsize_val = dbsize
                elif isinstance(dbsize, str):
                    dbsize_val = int(dbsize)
                else:
                    # Fallback: try converting via str()
                    dbsize_val = int(str(dbsize))
            except Exception:
                # If all conversions fail, preserve original value
                dbsize_val = dbsize

            if isinstance(info, dict):
                stats['redis'] = {
                    'connected_clients': info.get('connected_clients', 0),
                    'used_memory_human': info.get('used_memory_human', 'Unknown'),
                    'total_keys': dbsize_val,
                    'uptime_in_days': info.get('uptime_in_days', 0)
                }
            else:
                stats['redis_error'] = 'Redis info returned invalid format'
        except Exception as e:
            stats['redis_error'] = str(e)
    
    return jsonify(stats)

def warm_cache():
    """Pre-warm Redis cache from Supabase (useful after Redis restart)"""
    try:
        if not REDIS_AVAILABLE or not redis_client:
            return jsonify({'error': 'Redis not available'}), 503

        def _extract_rows(resp):
            """Robustly extract a list of rows from various Supabase response shapes."""
            if resp is None:
                return []
            # object-like with .data
            try:
                data_attr = getattr(resp, 'data', None)
                if isinstance(data_attr, list):
                    return data_attr
            except Exception:
                pass
            # dict-like response
            if isinstance(resp, dict):
                data = resp.get('data') or resp.get('items') or resp.get('rows') or []
                if isinstance(data, list):
                    return data
                # handle JSON-encoded string in 'data'
                if isinstance(data, str):
                    try:
                        parsed = json.loads(data)
                        if isinstance(parsed, list):
                            return parsed
                    except Exception:
                        pass
            # response might be a JSON string
            if isinstance(resp, str):
                try:
                    parsed = json.loads(resp)
                    if isinstance(parsed, dict) and isinstance(parsed.get('data'), list):
                        return parsed.get('data')
                    if isinstance(parsed, list):
                        return parsed
                except Exception:
                    pass
            return []

        # Get recent configurations from Supabase
        response = supabase.table('configs').select('*').order('created_at', desc=True).limit(100).execute()

        warmed = 0
        configs = _extract_rows(response) or []
        for config in configs:
            if not isinstance(config, dict):
                continue
            config_key = config.get('config_key')
            config_data = config.get('config_data')
            if config_key and config_data:
                try:
                    redis_client.setex(f"config:{config_key}", 86400, json.dumps(config_data))
                    warmed += 1
                except Exception as e:
                    logger.warning(f"Failed to set config in redis for {config_key}: {e}")

        # Get recent suggestions
        response = supabase.table('suggestions').select('*').order('created_at', desc=True).limit(100).execute()

        suggestions = _extract_rows(response)
        if suggestions:
            for suggestion in suggestions:
                if not isinstance(suggestion, dict):
                    continue
                config_key = suggestion.get('config_key')
                suggestion_text = suggestion.get('suggestion_text') or suggestion.get('suggestion')
                if config_key and suggestion_text:
                    try:
                        redis_client.setex(f"suggestion:{config_key}", 86400, suggestion_text)
                        warmed += 1
                    except Exception as e:
                        logger.warning(f"Failed to set suggestion in redis for {config_key}: {e}")
        else:
            logger.info("No suggestions returned from Supabase to warm into Redis")

        logger.info(f"Warmed {warmed} cache entries from Supabase to Redis")
        return jsonify({'message': f'Cache warmed with {warmed} entries', 'success': True})

    except Exception as e:
        logger.error(f"Cache warming error: {e}")
        return jsonify({'error': str(e), 'success': False}), 500

# Utility routes
@app.route('/debug')
def debug():
    status = {
        'flask': 'running',
        'supabase': get_storage_status(),
        'openrouter': False,
        'user_logged_in': bool(session.get('access_token'))
    }
    
    try:
        supabase.table("profiles").select("id").limit(1).execute()
        status['supabase'] = True
    except:
        status['supabase'] = False
    
    if OPENROUTER_API_KEY:
        try:
            response = requests.get('https://openrouter.ai/api/v1/models', 
                                  headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"}, 
                                  timeout=1000)
            status['openrouter'] = response.status_code == 200
        except:
            status['openrouter'] = False
    
    return jsonify(status)

    
def _create_placeholder() -> Path:
    """Create a grey “Image not available” placeholder and return its Path."""
    placeholder_path = Path("static/placeholder.png")
    if placeholder_path.exists():
        return placeholder_path

    img = Image.new("RGB", (600, 400), color="#e0e0e0")
    draw = ImageDraw.Draw(img)
    draw.text((50, 180), "Image not available", fill="#555555", font_size=36)
    placeholder_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(placeholder_path)
    logger.info(f"Created placeholder image at {placeholder_path}")
    return placeholder_path

def placeholder_filename() -> str:
    """Return the filename (relative to static) of the placeholder."""
    return _create_placeholder().name

# Create placeholder image
def create_placeholder_image():
    img = Image.new('RGB', (400, 300), color='lightgray')
    d = ImageDraw.Draw(img)
    d.text((100, 140), "Image not available", fill='black')
    placeholder_path = os.path.join('static', 'placeholder.png')
    os.makedirs('static', exist_ok=True)
    img.save(placeholder_path)

def get_placeholder_path():
    path = os.path.join('static', 'placeholder.png')
    if not os.path.exists(path):
        create_placeholder_image()
    return path

if __name__ == '__main__':
    app.run(debug=True, port=5000)