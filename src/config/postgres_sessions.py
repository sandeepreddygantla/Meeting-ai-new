"""
PostgreSQL-based session interface for Meetings AI application.
Replaces SQLite sessions when using PostgreSQL + pgvector.
"""
import os
import logging
import pickle
from datetime import datetime, timedelta
from uuid import uuid4
from flask.sessions import SessionInterface, SessionMixin
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class PostgresSession(dict, SessionMixin):
    """Custom session class for PostgreSQL storage."""
    pass


class PostgresSessionInterface(SessionInterface):
    """
    PostgreSQL-based session interface.
    Uses the existing PostgreSQL connection for session storage.
    """
    
    def __init__(self, postgres_manager):
        """
        Initialize session interface.
        
        Args:
            postgres_manager: PostgresManager instance
        """
        self.postgres_manager = postgres_manager
        self.session_cookie_name = 'session'
    
    def open_session(self, app, request):
        """
        Load session from PostgreSQL database.
        
        Args:
            app: Flask application instance
            request: Flask request object
            
        Returns:
            Session object
        """
        session = PostgresSession()
        
        try:
            sid = request.cookies.get(self.session_cookie_name)
            if not sid:
                logger.debug("No session ID in cookies, returning empty session")
                return session
            
            # Use PostgreSQL to validate session
            user_id = self.postgres_manager.validate_session(sid)
            if user_id:
                # Load user data into session
                user = self.postgres_manager.get_user_by_id(user_id)
                if user:
                    session.update({
                        'user_id': user.user_id,
                        'username': user.username,
                        'email': user.email,
                        'full_name': user.full_name,
                        'is_active': user.is_active,
                        'logged_in': True
                    })
                    logger.debug(f"Loaded session data for user: {user.username}")
            else:
                logger.debug(f"No valid session found for ID: {sid}")
                
        except Exception as e:
            logger.error(f"Session load error: {e}")
        
        return session
    
    def save_session(self, app, session, response):
        """
        Save session to PostgreSQL database.
        
        Args:
            app: Flask application instance
            session: Session object
            response: Flask response object
        """
        try:
            domain = self.get_cookie_domain(app)
            path = self.get_cookie_path(app)
            
            # Handle empty or unmodified sessions
            if not session or not getattr(session, 'modified', True):
                if hasattr(session, 'modified') and session.modified:
                    response.delete_cookie(
                        self.session_cookie_name, 
                        domain=domain, 
                        path=path
                    )
                return
            
            # Get or generate session ID
            sid = None
            try:
                from flask import request as flask_request
                if flask_request:
                    sid = flask_request.cookies.get(self.session_cookie_name)
            except:
                pass
                
            if not sid:
                sid = str(uuid4())
            
            # Calculate expiry
            expiry = datetime.now() + app.permanent_session_lifetime
            
            # Save to PostgreSQL if user is logged in
            user_id = session.get('user_id')
            if user_id:
                success = self.postgres_manager.create_session(user_id, sid, expiry)
                if not success:
                    logger.error(f"Failed to save session for user: {user_id}")
                    return
            
            # Set cookie
            response.set_cookie(
                self.session_cookie_name, 
                sid,
                expires=expiry, 
                httponly=True,
                domain=domain, 
                path=path, 
                secure=app.config.get('SESSION_COOKIE_SECURE', False)
            )
            
            logger.debug(f"Session saved successfully with ID: {sid}")
            
        except Exception as e:
            logger.error(f"Session save error: {e}")
    
    def get_cookie_name(self, app):
        """Get the session cookie name."""
        return self.session_cookie_name


def setup_postgres_session(app, postgres_manager):
    """
    Setup PostgreSQL-backed session interface for Flask app.
    
    Args:
        app: Flask application instance
        postgres_manager: PostgresManager instance
    """
    app.session_interface = PostgresSessionInterface(postgres_manager)
    return app.session_interface