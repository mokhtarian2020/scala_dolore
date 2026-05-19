import sqlite3
import json
import base64
from datetime import datetime
from typing import Optional, Dict, Any
import logging

class WelodgeConnector:
    """
    Database connector for Welodge system to manage patient data and reference images
    """
    
    def __init__(self, db_path: str = "welodge.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables if they don't exist"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS patient_references (
                    patient_id TEXT PRIMARY KEY,
                    reference_image BLOB,
                    creation_date TIMESTAMP,
                    last_updated TIMESTAMP,
                    metadata TEXT
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS pain_assessments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    patient_id TEXT,
                    target_image BLOB,
                    reference_image BLOB,
                    pain_score REAL,
                    assessment_date TIMESTAMP,
                    metadata TEXT,
                    FOREIGN KEY (patient_id) REFERENCES patient_references (patient_id)
                )
            ''')
    
    def has_reference_image(self, patient_id: str) -> bool:
        """Check if patient has a reference image"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT 1 FROM patient_references WHERE patient_id = ?", 
                (patient_id,)
            )
            return cursor.fetchone() is not None
    
    def save_reference_image(self, patient_id: str, image_data: bytes, metadata: Dict = None):
        """Save or update reference image for patient"""
        now = datetime.now()
        metadata_json = json.dumps(metadata or {})
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO patient_references 
                (patient_id, reference_image, creation_date, last_updated, metadata)
                VALUES (?, ?, ?, ?, ?)
            ''', (patient_id, image_data, now, now, metadata_json))
    
    def get_reference_image(self, patient_id: str) -> Optional[bytes]:
        """Retrieve reference image for patient"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT reference_image FROM patient_references WHERE patient_id = ?",
                (patient_id,)
            )
            result = cursor.fetchone()
            return result[0] if result else None
    
    def save_assessment(self, patient_id: str, target_image: bytes, 
                       reference_image: bytes, pain_score: float, 
                       metadata: Dict = None):
        """Save pain assessment results"""
        now = datetime.now()
        metadata_json = json.dumps(metadata or {})
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO pain_assessments 
                (patient_id, target_image, reference_image, pain_score, assessment_date, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (patient_id, target_image, reference_image, pain_score, now, metadata_json))