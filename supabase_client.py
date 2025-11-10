import os
import json
import logging
from typing import Dict, Optional, Any
import redis
from datetime import datetime, time
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Debug print to verify loading
print("SUPABASE_URL:", os.getenv("SUPABASE_URL"))
print("SUPABASE_ANON_KEY:", "Found" if os.getenv("SUPABASE_ANON_KEY") else "Missing")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("SUPABASE_URL or SUPABASE_ANON_KEY not found. Check your .env file.")


supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
print("✅ Connected to Supabase successfully!")

# Force schema cache refresh
try:
    print("Refreshing schema cache...")
    # This query will force Supabase to update its schema cache
    supabase.table('visitors').select('id').limit(1).execute()
    print("✅ Schema cache refreshed")
except Exception as e:
    print(f"⚠️ Schema refresh warning: {e}")
# Initialize Redis client
try:
    redis_client = redis.Redis(
        host=os.getenv('REDIS_HOST', 'localhost'),
        port=int(os.getenv('REDIS_PORT', 6379)),
        db=0,
        decode_responses=True,
        socket_connect_timeout=5
    )
    redis_client.ping()
    logger.info("Connected to Redis successfully")
    REDIS_AVAILABLE = True
except (redis.ConnectionError, redis.TimeoutError) as e:
    logger.warning(f"Redis connection failed: {e}. Falling back to Supabase only.")
    redis_client = None
    REDIS_AVAILABLE = False

# Cache TTL settings (in seconds)
REDIS_CONFIG_TTL = 86400  # 24 hours
REDIS_SUGGESTION_TTL = 86400  # 24 hours
REDIS_CHAT_TTL = 3600  # 1 hour

# ============================================
# Configuration Storage (Redis Cache + Supabase Persistent)
# ============================================

def store_config(config_key: str, config_data: Dict[str, Any]) -> bool:
    try:
        config_json = json.dumps(config_data)
        
        # Store in Supabase (permanent storage)
        supabase_data = {
            'config_key': config_key,
            'config_data': config_data,
            'user_id': config_data.get('user_id')
        }
        supabase.table('configs').upsert(supabase_data, on_conflict='config_key').execute()
        logger.debug(f"Stored config in Supabase for key: {config_key}")
        
        # Cache in Redis for fast access
        if REDIS_AVAILABLE and redis_client:
            redis_client.setex(f"config:{config_key}", REDIS_CONFIG_TTL, config_json)
            logger.debug(f"Cached config in Redis for key: {config_key}")
        
        return True
    except Exception as e:
        logger.error(f"Error storing config: {e}")
        return False

def get_config(config_key: str) -> Optional[Dict[str, Any]]:
    """
    Get configuration from Redis cache first, fallback to Supabase
    Strategy: Check Redis → if miss, fetch from Supabase and cache
    """
    try:
        # Try Redis cache first (fast path)
        if REDIS_AVAILABLE and redis_client:
            cached = redis_client.get(f"config:{config_key}")
            if cached:
                logger.debug(f"Config cache HIT in Redis for key: {config_key}")
                return json.loads(cached)
            logger.debug(f"Config cache MISS in Redis for key: {config_key}")
        
        # Fallback to Supabase (slow path)
        response = supabase.table('configs').select('*').eq('config_key', config_key).execute()
        
        if response.data and len(response.data) > 0:
            config_data = response.data[0].get('config_data')
            logger.debug(f"Config retrieved from Supabase for key: {config_key}")
            
            # Cache in Redis for next time
            if REDIS_AVAILABLE and redis_client and config_data:
                redis_client.setex(f"config:{config_key}", REDIS_CONFIG_TTL, json.dumps(config_data))
                logger.debug(f"Cached config from Supabase to Redis: {config_key}")
            
            return config_data
        
        return None
    except Exception as e:
        logger.error(f"Error retrieving config: {e}")
        return None

# ============================================
# Suggestion Cache (Redis Cache + Supabase Backup)
# ============================================

def store_suggestion(config_key: str, suggestion_text: str) -> bool:
    """
    Store AI suggestion in both Redis (cache) and Supabase (backup)
    Strategy: Redis for speed, Supabase for history
    """
    try:
        # Store in Supabase (permanent storage)
        data = {
            'config_key': config_key,
            'suggestion_text': suggestion_text,
            'created_at': datetime.utcnow().isoformat()
        }
        supabase.table('suggestions').upsert(data, on_conflict='config_key').execute()
        logger.debug(f"Stored suggestion in Supabase for key: {config_key}")
        
        # Cache in Redis for fast access
        if REDIS_AVAILABLE and redis_client:
            redis_client.setex(f"suggestion:{config_key}", REDIS_SUGGESTION_TTL, suggestion_text)
            logger.debug(f"Cached suggestion in Redis for key: {config_key}")
        
        return True
    except Exception as e:
        logger.error(f"Error storing suggestion: {e}")
        return False

def get_suggestion(config_key: str) -> Optional[str]:
    """
    Get suggestion from Redis cache first, fallback to Supabase
    Strategy: Check Redis → if miss, fetch from Supabase and cache
    """
    try:
        # Try Redis cache first (fast path)
        if REDIS_AVAILABLE and redis_client:
            cached = redis_client.get(f"suggestion:{config_key}")
            if cached:
                logger.debug(f"Suggestion cache HIT in Redis for key: {config_key}")
                return cached
            logger.debug(f"Suggestion cache MISS in Redis for key: {config_key}")
        
        # Fallback to Supabase (slow path)
        response = supabase.table('suggestions').select('*').eq('config_key', config_key).execute()
        
        if response.data and len(response.data) > 0:
            suggestion_text = response.data[0].get('suggestion_text')
            logger.debug(f"Suggestion retrieved from Supabase for key: {config_key}")
            
            # Cache in Redis for next time
            if REDIS_AVAILABLE and redis_client and suggestion_text:
                redis_client.setex(f"suggestion:{config_key}", REDIS_SUGGESTION_TTL, suggestion_text)
                logger.debug(f"Cached suggestion from Supabase to Redis: {config_key}")
            
            return suggestion_text
        
        return None
    except Exception as e:
        logger.error(f"Error retrieving suggestion: {e}")
        return None

# ============================================
# Chat Cache (Redis Only - Temporary Data)
# ============================================

def store_chat_cache(cache_key: str, response_text: str) -> bool:
    """
    Store chat response in Redis only (temporary cache)
    Strategy: Redis only - chat cache doesn't need persistence
    """
    try:
        if REDIS_AVAILABLE and redis_client:
            redis_client.setex(cache_key, REDIS_CHAT_TTL, response_text)
            logger.debug(f"Cached chat response in Redis for key: {cache_key}")
            return True
        else:
            # Fallback: Store in Supabase if Redis unavailable
            data = {
                'cache_key': cache_key,
                'response_text': response_text,
                'created_at': datetime.utcnow().isoformat()
            }
            supabase.table('chat_cache').upsert(data, on_conflict='cache_key').execute()
            logger.debug(f"Stored chat cache in Supabase (Redis unavailable): {cache_key}")
            return True
    except Exception as e:
        logger.error(f"Error storing chat cache: {e}")
        return False

def get_chat_cache(cache_key: str) -> Optional[str]:
    """
    Get chat response from Redis cache
    Strategy: Redis only for speed - short TTL, no need for Supabase
    """
    try:
        if REDIS_AVAILABLE and redis_client:
            cached = redis_client.get(cache_key)
            if cached:
                logger.debug(f"Chat cache HIT in Redis for key: {cache_key}")
                return cached
            logger.debug(f"Chat cache MISS in Redis for key: {cache_key}")
        else:
            # Fallback: Check Supabase if Redis unavailable
            response = supabase.table('chat_cache').select('*').eq('cache_key', cache_key).execute()
            if response.data and len(response.data) > 0:
                logger.debug(f"Chat cache retrieved from Supabase: {cache_key}")
                return response.data[0].get('response_text')
        
        return None
    except Exception as e:
        logger.error(f"Error retrieving chat cache: {e}")
        return None

# ============================================
# Cache Management
# ============================================

def clear_all_cache() -> bool:
    """
    Clear all cached data from both Redis and Supabase
    Strategy: Clear Redis immediately, optionally clear old Supabase records
    """
    try:
        # Clear Redis cache
        if REDIS_AVAILABLE and redis_client:
            redis_client.flushdb()
            logger.info("Redis cache cleared successfully")
        
        # Optionally clear Supabase cache tables (keep permanent data)
        # Only clear chat_cache from Supabase (temporary data)
        supabase.table('chat_cache').delete().neq('cache_key', '').execute()
        logger.info("Supabase chat cache cleared successfully")
        
        return True
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        return False

def clear_redis_only() -> bool:
    """Clear only Redis cache, keep Supabase data intact"""
    try:
        if REDIS_AVAILABLE and redis_client:
            redis_client.flushdb()
            logger.info("Redis cache cleared (Supabase data preserved)")
            return True
        return False
    except Exception as e:
        logger.error(f"Error clearing Redis cache: {e}")
        return False

# ============================================
# User Events & Analytics (Supabase Only - Permanent Data)
# ============================================

def log_user_event(user_id: str, event_type: str, event_data: Optional[Dict[str, Any]] = None) -> bool:
    """
    Log user events for analytics (Supabase only)
    Strategy: Permanent data - no caching needed
    """
    try:
        data = {
            'user_id': user_id,
            'event_type': event_type,
            'event_data': json.dumps(event_data) if event_data else None,
            'created_at': datetime.utcnow().isoformat()
        }
        
        supabase.table('user_events').insert(data).execute()
        logger.debug(f"Logged event: {event_type} for user: {user_id}")
        return True
    except Exception as e:
        logger.error(f"Error logging user event: {e}")
        return False

def log_visitor(ip_address: str) -> bool:
    """
    Log visitor access (Supabase only)
    Strategy: Permanent data - no caching needed
    """
    max_retries = 2
    
    for attempt in range(max_retries):
        try:
            data = {
                'ip_address': ip_address,
                'visited_at': datetime.utcnow().isoformat()
            }
            
            response = supabase.table('visitors').insert(data).execute()
            logger.debug(f"Logged visitor: {ip_address}")
            return True
            
        except Exception as e:
            logger.error(f"Error logging visitor (attempt {attempt + 1}): {e}")
            
            # If it's a schema cache error, wait and retry
            if 'PGRST204' in str(e) or 'schema cache' in str(e).lower():
                if attempt < max_retries - 1:
                    logger.info("Schema cache issue detected, waiting before retry...")
                    time.sleep(1)
                    continue
            # For other errors or final attempt failure, just log and continue
            break
    
    logger.warning(f"Failed to log visitor after {max_retries} attempts: {ip_address}")
    return False

# ============================================
# Health Check Functions
# ============================================

def check_redis_health() -> bool:
    """Check if Redis is available"""
    if not REDIS_AVAILABLE or not redis_client:
        return False
    try:
        redis_client.ping()
        return True
    except:
        return False

def check_supabase_health() -> bool:
    """Check if Supabase is available"""
    try:
        supabase.table('profiles').select('id', count='exact').limit(1).execute()
        return True
    except:
        return False

def get_storage_status() -> Dict[str, Any]:
    """Get status of both storage systems"""
    return {
        'redis': {
            'available': REDIS_AVAILABLE,
            'connected': check_redis_health()
        },
        'supabase': {
            'available': True,
            'connected': check_supabase_health()
        }
    }

