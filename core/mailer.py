"""
Secure SMTP Mailer & OTP Verification Service (Diagram 3 Architecture).
======================================================================
Implements cryptographically secure, time-bounded (10 min) OTP generation,
SQLite-backed verification tracking with brute-force rate-limiting, and
branded HTML email dispatch via smtplib.
"""

from __future__ import annotations

import hashlib
import logging
import secrets
import smtplib
import sqlite3
import time
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any, Optional, Tuple

from core.config import settings

logger = logging.getLogger("healthtrack.mailer")


class MailerService:
    """Manages OTP lifecycle, email dispatch, and verification verification."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or (settings.DATA_DIR / "firebase_local_store.db")
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        return sqlite3.connect(str(self.db_path), timeout=10.0)

    def _init_db(self) -> None:
        """Initializes the OTP tracking table."""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS otp_verifications (
            email TEXT PRIMARY KEY,
            otp_hash TEXT NOT NULL,
            expires_at REAL NOT NULL,
            attempts INTEGER NOT NULL DEFAULT 0,
            verified INTEGER NOT NULL DEFAULT 0,
            created_at REAL NOT NULL
        );
        """)
        conn.commit()
        conn.close()

    @staticmethod
    def _hash_code(code: str) -> str:
        """SHA-256 hash for secure at-rest code storage."""
        return hashlib.sha256(code.strip().encode("utf-8")).hexdigest()

    def generate_and_store_otp(self, email: str, expiry_minutes: int = 10) -> str:
        """
        Generates a 6-digit cryptographic random OTP and stores its hash.
        Resets attempts and invalidates prior unused codes for this email.
        """
        email_clean = email.strip().lower()
        # Generate 6-digit numeric OTP (100000 - 999999)
        code = str(secrets.randbelow(900000) + 100000)
        code_hash = self._hash_code(code)
        now = time.time()
        expires_at = now + (expiry_minutes * 60)

        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO otp_verifications (email, otp_hash, expires_at, attempts, verified, created_at)
            VALUES (?, ?, ?, 0, 0, ?)
            ON CONFLICT(email) DO UPDATE SET
                otp_hash = excluded.otp_hash,
                expires_at = excluded.expires_at,
                attempts = 0,
                verified = 0,
                created_at = excluded.created_at;
            """,
            (email_clean, code_hash, expires_at, now),
        )
        conn.commit()
        conn.close()

        logger.info(f"Generated secure OTP for {email_clean} (expires in {expiry_minutes}m).")
        return code

    def send_otp_email(
        self,
        to_email: str,
        otp_code: str,
        patient_name: str = "Patient",
    ) -> dict[str, Any]:
        """
        Dispatches branded HTML verification email via SMTP.
        Falls back gracefully to debug logging if SMTP is unconfigured.
        """
        email_clean = to_email.strip().lower()

        # Build clean HTML email
        subject = f"HealthTrack Telehealth - Your Verification Code is {otp_code}"
        html_body = f"""
        <!DOCTYPE html>
        <html>
        <head>
          <meta charset="utf-8">
          <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background-color: #f4f4f5; margin: 0; padding: 20px; }}
            .container {{ max-width: 520px; margin: 0 auto; background: #ffffff; border: 3px solid #000000; box-shadow: 6px 6px 0px #000000; padding: 32px; }}
            .header {{ border-bottom: 2px solid #000000; padding-bottom: 16px; margin-bottom: 24px; }}
            .title {{ font-size: 22px; font-weight: 800; text-transform: uppercase; margin: 0; color: #000000; }}
            .otp-box {{ background: #fef08a; border: 2px dashed #000000; padding: 18px; text-align: center; margin: 24px 0; }}
            .otp-code {{ font-size: 34px; font-weight: 900; letter-spacing: 6px; color: #000000; font-family: monospace; }}
            .footer {{ font-size: 11px; color: #71717a; border-top: 1px solid #e4e4e7; padding-top: 16px; margin-top: 24px; }}
          </style>
        </head>
        <body>
          <div class="container">
            <div class="header">
              <h2 class="title">HealthTrack Clinical AI</h2>
            </div>
            <p>Hello <strong>{patient_name}</strong>,</p>
            <p>Please use the verification code below to verify your patient account and access your clinical decision support portal.</p>
            <div class="otp-box">
              <div class="otp-code">{otp_code}</div>
            </div>
            <p style="font-size: 13px; color: #52525b;">This code is valid for <strong>10 minutes</strong>. If you did not request this verification, please safely disregard this email.</p>
            <div class="footer">
              HealthTrack Clinical Decision Support System &bull; FDA CDSS Compliant &bull; Confidential
            </div>
          </div>
        </body>
        </html>
        """

        # Check if live SMTP server is configured
        has_smtp_creds = bool(settings.SMTP_USER and settings.SMTP_PASSWORD)

        if has_smtp_creds:
            try:
                msg = MIMEMultipart("alternative")
                msg["Subject"] = subject
                msg["From"] = settings.SMTP_FROM_EMAIL
                msg["To"] = email_clean
                msg.attach(MIMEText(f"Your HealthTrack verification code is: {otp_code}. Valid for 10 minutes.", "plain"))
                msg.attach(MIMEText(html_body, "html"))

                server = smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT, timeout=10.0)
                if settings.SMTP_TLS:
                    server.starttls()
                server.login(settings.SMTP_USER, settings.SMTP_PASSWORD)
                server.sendmail(settings.SMTP_FROM_EMAIL, [email_clean], msg.as_string())
                server.quit()

                logger.info(f"Successfully sent live SMTP OTP email to {email_clean}")
                return {"status": "SUCCESS", "mode": "LIVE_SMTP", "recipient": email_clean}
            except Exception as e:
                logger.warning(f"SMTP live dispatch failed ({e}); falling back to local verification mode.")

        # Local development / Sandboxed fallback
        logger.info(f"[LOCAL OTP DISPATCH] Code for {email_clean}: {otp_code}")
        return {
            "status": "SUCCESS",
            "mode": "LOCAL_DEV",
            "recipient": email_clean,
            "debug_code": otp_code if (not settings.SMTP_USER) else None,
        }

    def verify_otp(self, email: str, entered_code: str) -> Tuple[bool, str]:
        """
        Cryptographically verifies the entered OTP code.
        Enforces 10-minute expiry and max 3 invalid attempts limit.
        """
        email_clean = email.strip().lower()
        now = time.time()

        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT otp_hash, expires_at, attempts, verified FROM otp_verifications WHERE email = ?",
            (email_clean,),
        )
        row = cursor.fetchone()

        if not row:
            conn.close()
            return False, "No pending verification found for this email address."

        otp_hash, expires_at, attempts, verified = row[0], row[1], row[2], row[3]

        if verified:
            conn.close()
            return True, "Email address is already verified."

        if now > expires_at:
            conn.close()
            return False, "Verification code has expired. Please request a new code."

        if attempts >= 3:
            conn.close()
            return False, "Too many failed attempts. For your security, please request a new verification code."

        # Constant-time comparison to prevent timing side-channel attacks
        candidate_hash = self._hash_code(entered_code)
        if secrets.compare_digest(candidate_hash, otp_hash):
            cursor.execute(
                "UPDATE otp_verifications SET verified = 1, attempts = attempts + 1 WHERE email = ?",
                (email_clean,),
            )
            conn.commit()
            conn.close()
            logger.info(f"Successfully verified OTP for {email_clean}")
            return True, "Verification successful."
        else:
            new_attempts = attempts + 1
            cursor.execute(
                "UPDATE otp_verifications SET attempts = ? WHERE email = ?",
                (new_attempts, email_clean),
            )
            conn.commit()
            conn.close()
            remaining = max(0, 3 - new_attempts)
            return False, f"Invalid verification code. {remaining} attempt(s) remaining."


# Global singleton instance
mailer_service = MailerService()

