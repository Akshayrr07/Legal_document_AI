"""
History API route — retrieves past analysis records for a given user.
"""

import logging
from flask import Blueprint, jsonify, request
from utils.db import Database
from backend.config import Config

logger = logging.getLogger(__name__)

history_bp = Blueprint("history", __name__)

# Shared DB instance (SQLite connection is per-request internally)
_db: Database = Database(db_path=Config.DATABASE_PATH)


@history_bp.route("/history", methods=["GET"])
def get_history():
    """
    GET /api/history?user_id=<id>

    Returns a list of past analysis records for the specified user,
    ordered by most recent first.
    """
    user_id: str | None = request.args.get("user_id", "").strip() or None

    if not user_id:
        return jsonify({"error": "Query parameter 'user_id' is required."}), 400

    try:
        history = _db.fetch_user_history(user_id)
        logger.debug("Fetched %d history records for user '%s'.", len(history), user_id)
    except Exception as exc:
        logger.error("Database error fetching history for user '%s': %s", user_id, exc)
        return jsonify({"error": "Failed to retrieve history.", "detail": str(exc)}), 500

    return jsonify({"history": history}), 200
