from flask import Flask, request, jsonify
from datetime import datetime
import os

# Import Supabase helper functions
from supabase_client import (
    supabase,
    log_visitor,
    get_dashboard_stats
)

app = Flask(__name__)

@app.route("/")
def dashboard():
    """
    Dashboard endpoint that provides analytics data.
    All data is now retrieved from Supabase instead of SQLite.
    """
    # Log the visitor
    log_visitor(request.remote_addr)
    
    # Get all dashboard statistics from Supabase
    stats = get_dashboard_stats()
    
    return jsonify(stats)

@app.route("/health")
def health():
    """Health check endpoint"""
    try:
        # Test Supabase connection
        supabase.table("profiles").select("id", count="exact").limit(1).execute()
        return jsonify({
            'status': 'healthy',
            'database': 'supabase',
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route("/stats/users")
def user_stats():
    """Get detailed user statistics"""
    try:
        # Total users
        users_response = supabase.table("profiles").select("id", count="exact").execute()
        total_users = users_response.count if users_response.count else 0
        
        # Get all user events
        events_response = supabase.table("user_events").select("*").order("event_time", desc=True).execute()
        
        return jsonify({
            'total_users': total_users,
            'recent_events': events_response.data[:10] if events_response.data else []
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route("/stats/visitors")
def visitor_stats():
    """Get visitor statistics"""
    try:
        # Total visitors
        visitors_response = supabase.table("visitors").select("id", count="exact").execute()
        total_visitors = visitors_response.count if visitors_response.count else 0
        
        # Recent visitors
        recent_response = supabase.table("visitors").select("*").order("visited_at", desc=True).limit(20).execute()
        
        return jsonify({
            'total_visitors': total_visitors,
            'recent_visitors': recent_response.data if recent_response.data else []
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route("/stats/activity")
def activity_stats():
    """Get activity statistics by date"""
    try:
        # Get new users per day
        new_users_response = supabase.rpc("get_new_users_per_day").execute()
        new_users = {row['day']: row['count'] for row in (new_users_response.data or [])}
        
        # Get deleted users per day
        deleted_users_response = supabase.rpc("get_deleted_users_per_day").execute()
        deleted_users = {row['day']: row['count'] for row in (deleted_users_response.data or [])}
        
        # Get visits per day
        visits_response = supabase.rpc("get_visits_per_day").execute()
        visits = {row['day']: row['count'] for row in (visits_response.data or [])}
        
        return jsonify({
            'new_users_per_day': new_users,
            'deleted_users_per_day': deleted_users,
            'visits_per_day': visits
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route("/debug")
def debug():
    """Debug endpoint to check Supabase connection"""
    try:
        # Test connection
        response = supabase.table("profiles").select("id", count="exact").limit(1).execute()
        
        return jsonify({
            'supabase_connected': True,
            'database_type': 'Supabase PostgreSQL',
            'test_query_success': True,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'supabase_connected': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

# Run the app
if __name__ == '__main__':
    print("=" * 50)
    print("Dashboard Service (Supabase Edition)")
    print("=" * 50)
    print("Using Supabase as database (no SQLite)")
    print("Endpoints:")
    print("  - GET /          : Full dashboard stats")
    print("  - GET /health    : Health check")
    print("  - GET /stats/users    : User statistics")
    print("  - GET /stats/visitors : Visitor statistics")
    print("  - GET /stats/activity : Activity by date")
    print("  - GET /debug     : Debug information")
    print("=" * 50)
    app.run(debug=True, use_reloader=False, port=5002)