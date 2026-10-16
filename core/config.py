"""Application configuration and environment settings."""

import os
from pathlib import Path
from pydantic import BaseModel

# Pre-load .env into environment so all settings have direct access to variables
_project_root = Path(__file__).resolve().parent.parent
_env_path = _project_root / ".env"
if _env_path.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=_env_path, override=False)
    except ImportError:
        pass
    try:
        with open(_env_path, "r", encoding="utf-8") as _f:
            for _line in _f:
                _line = _line.strip()
                if _line and not _line.startswith("#") and "=" in _line:
                    _k, _v = _line.split("=", 1)
                    _k = _k.strip()
                    _v = _v.strip().strip("'\"")
                    if _k and _k not in os.environ:
                        os.environ[_k] = _v
    except Exception:
        pass


def _resolve_firebase_cred_path(root: Path) -> Path:
    raw = os.getenv("FIREBASE_CREDENTIALS_PATH", "").strip()
    if not raw:
        return root / "firebase-credentials.json"
    p = Path(raw)
    return p if p.is_absolute() else (root / p)


try:
    from pydantic_settings import BaseSettings, SettingsConfigDict

    class Settings(BaseSettings):
        """Production settings with environment variable support."""

        PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
        DATA_DIR: Path = PROJECT_ROOT / "data"
        DATASETS_DIR: Path = PROJECT_ROOT / "datasets"
        CHECKPOINTS_DIR: Path = PROJECT_ROOT / "checkpoints"
        EXPERIMENTS_DIR: Path = PROJECT_ROOT / "experiments"

        OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "qwen2.5:0.5b")
        FALLBACK_MODEL: str = os.getenv("FALLBACK_MODEL", "qwen2.5:0.5b")
        OLLAMA_TIMEOUT_SECONDS: float = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "60.0"))

        # Medical Knowledge & Scraper Settings
        MEDLINEPLUS_API_ENABLED: bool = True
        PUBMED_API_ENABLED: bool = True
        OPENROUTER_API_KEY: str | None = os.getenv("OPENROUTER_API_KEY", None)
        NVIDIA_NIM_API_KEY: str | None = os.getenv("NVIDIA_NIM_API_KEY", None)

        APP_HOST: str = os.getenv("APP_HOST", "0.0.0.0")
        APP_PORT: int = int(os.getenv("APP_PORT", "8000"))
        CLOUDFLARE_TUNNEL_PORT: int = int(os.getenv("CLOUDFLARE_TUNNEL_PORT", "8000"))

        LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

        ANCHOR_DATA_RATIO: float = 0.70
        NEW_WEB_DATA_RATIO: float = 0.30

        MIN_CITATION_SCORE: float = 0.85
        MAX_REGRESSION_DELTA: float = 0.05

        # Firebase Credentials — strictly loaded from environment / .env
        FIREBASE_CREDENTIALS_PATH: Path = _resolve_firebase_cred_path(PROJECT_ROOT)
        FIREBASE_PROJECT_ID: str = os.getenv("FIREBASE_PROJECT_ID", "")
        FIREBASE_STORAGE_BUCKET: str = os.getenv("FIREBASE_STORAGE_BUCKET", "")
        FIREBASE_API_KEY: str = os.getenv("FIREBASE_API_KEY", "")
        FIREBASE_AUTH_DOMAIN: str = os.getenv("FIREBASE_AUTH_DOMAIN", "")
        FIREBASE_MESSAGING_SENDER_ID: str = os.getenv("FIREBASE_MESSAGING_SENDER_ID", "")
        FIREBASE_APP_ID: str = os.getenv("FIREBASE_APP_ID", "")
        FIREBASE_MEASUREMENT_ID: str = os.getenv("FIREBASE_MEASUREMENT_ID", "")

        # SMTP & OTP Verification Configuration (Diagram 3)
        SMTP_HOST: str = os.getenv("SMTP_HOST", "smtp.gmail.com")
        SMTP_PORT: int = int(os.getenv("SMTP_PORT", "587"))
        SMTP_USER: str = os.getenv("SMTP_USER", "")
        SMTP_PASSWORD: str = os.getenv("SMTP_PASSWORD", "")
        SMTP_FROM_EMAIL: str = os.getenv("SMTP_FROM_EMAIL", "noreply@healthtrack.ai")
        SMTP_TLS: bool = os.getenv("SMTP_TLS", "true").lower() == "true"

        model_config = SettingsConfigDict(
            env_file=".env",
            env_file_encoding="utf-8",
            extra="ignore",
        )

except ImportError:
    # Graceful fallback when pydantic-settings is not yet installed in local environment
    class Settings(BaseModel):  # type: ignore[no-redef]
        """Fallback settings reading directly from os.getenv."""

        PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
        DATA_DIR: Path = PROJECT_ROOT / "data"
        DATASETS_DIR: Path = PROJECT_ROOT / "datasets"
        CHECKPOINTS_DIR: Path = PROJECT_ROOT / "checkpoints"
        EXPERIMENTS_DIR: Path = PROJECT_ROOT / "experiments"

        OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "qwen2.5:0.5b")
        FALLBACK_MODEL: str = os.getenv("FALLBACK_MODEL", "qwen2.5:0.5b")
        OLLAMA_TIMEOUT_SECONDS: float = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "60.0"))

        MEDLINEPLUS_API_ENABLED: bool = os.getenv("MEDLINEPLUS_API_ENABLED", "true").lower() == "true"
        PUBMED_API_ENABLED: bool = os.getenv("PUBMED_API_ENABLED", "true").lower() == "true"
        OPENROUTER_API_KEY: str | None = os.getenv("OPENROUTER_API_KEY", None)
        NVIDIA_NIM_API_KEY: str | None = os.getenv("NVIDIA_NIM_API_KEY", None)

        APP_HOST: str = os.getenv("APP_HOST", "0.0.0.0")
        APP_PORT: int = int(os.getenv("APP_PORT", "8000"))
        CLOUDFLARE_TUNNEL_PORT: int = int(os.getenv("CLOUDFLARE_TUNNEL_PORT", "8000"))

        LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

        ANCHOR_DATA_RATIO: float = float(os.getenv("ANCHOR_DATA_RATIO", "0.70"))
        NEW_WEB_DATA_RATIO: float = float(os.getenv("NEW_WEB_DATA_RATIO", "0.30"))

        MIN_CITATION_SCORE: float = float(os.getenv("MIN_CITATION_SCORE", "0.85"))
        MAX_REGRESSION_DELTA: float = float(os.getenv("MAX_REGRESSION_DELTA", "0.05"))

        # Firebase Credentials — strictly loaded from environment / .env
        FIREBASE_CREDENTIALS_PATH: Path = _resolve_firebase_cred_path(PROJECT_ROOT)
        FIREBASE_PROJECT_ID: str = os.getenv("FIREBASE_PROJECT_ID", "")
        FIREBASE_STORAGE_BUCKET: str = os.getenv("FIREBASE_STORAGE_BUCKET", "")
        FIREBASE_API_KEY: str = os.getenv("FIREBASE_API_KEY", "")
        FIREBASE_AUTH_DOMAIN: str = os.getenv("FIREBASE_AUTH_DOMAIN", "")
        FIREBASE_MESSAGING_SENDER_ID: str = os.getenv("FIREBASE_MESSAGING_SENDER_ID", "")
        FIREBASE_APP_ID: str = os.getenv("FIREBASE_APP_ID", "")
        FIREBASE_MEASUREMENT_ID: str = os.getenv("FIREBASE_MEASUREMENT_ID", "")

        # SMTP & OTP Verification Configuration (Diagram 3)
        SMTP_HOST: str = os.getenv("SMTP_HOST", "smtp.gmail.com")
        SMTP_PORT: int = int(os.getenv("SMTP_PORT", "587"))
        SMTP_USER: str = os.getenv("SMTP_USER", "")
        SMTP_PASSWORD: str = os.getenv("SMTP_PASSWORD", "")
        SMTP_FROM_EMAIL: str = os.getenv("SMTP_FROM_EMAIL", "noreply@healthtrack.ai")
        SMTP_TLS: bool = os.getenv("SMTP_TLS", "true").lower() == "true"


settings = Settings()

# Ensure directories exist
for path in [
    settings.DATA_DIR,
    settings.DATASETS_DIR,
    settings.CHECKPOINTS_DIR,
    settings.EXPERIMENTS_DIR,
]:
    path.mkdir(parents=True, exist_ok=True)
