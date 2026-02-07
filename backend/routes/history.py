import os
from flask import Blueprint, jsonify, request
from utils.db import Database

history_bp = Blueprint("history", __name__)

db = Database(
    db_path=os.getenv("DATABASE_PATH")
)


@history_bp.route("/history", methods=["GET"])
def get_history():
    """
    Fetch past analyses for a user.
    User ID is passed as query param (session auth later).
    """
    user_id = request.args.get("user_id")

    if not user_id:
        return jsonify({"error": "user_id is required"}), 400

    history = db.fetch_user_history(user_id)
    return jsonify({"history": history}), 200
