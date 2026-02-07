import sqlite3
import json
from datetime import datetime
from typing import Dict, List


class Database:
    """
    SQLite persistence layer for analyzed legal documents.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS analyses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    document_name TEXT NOT NULL,
                    ocr_confidence REAL,
                    summary TEXT,
                    risk_analysis TEXT,
                    created_at TEXT
                )
            """)
            conn.commit()

    def save_result(
        self,
        user_id: str,
        document_name: str,
        ocr_confidence: float,
        summary: str,
        risk_analysis: Dict
    ):
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO analyses (
                    user_id, document_name, ocr_confidence,
                    summary, risk_analysis, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                user_id,
                document_name,
                ocr_confidence,
                summary,
                json.dumps(risk_analysis),
                datetime.utcnow().isoformat()
            ))
            conn.commit()

    def fetch_user_history(self, user_id: str) -> List[Dict]:
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT document_name, ocr_confidence, summary,
                       risk_analysis, created_at
                FROM analyses
                WHERE user_id = ?
                ORDER BY created_at DESC
            """, (user_id,))

            rows = cursor.fetchall()

        results = []
        for row in rows:
            results.append({
                "document_name": row[0],
                "ocr_confidence": row[1],
                "summary": row[2],
                "risk_analysis": json.loads(row[3]),
                "created_at": row[4]
            })

        return results
