from pathlib import Path
from typing import List, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AGRIVISION_",
        extra="ignore",
    )

    PROJECT_NAME: str = "AgriVision AI Backend"
    VERSION: str = "1.0.0"
    API_PREFIX: str = "/api"

    # Base project root resolved dynamically relative to this configuration file
    PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]

    # Optional overrides for file paths
    METRICS_FILE_PATH: Optional[Path] = None
    OPERATIONAL_STATUS_FILE_PATH: Optional[Path] = None
    REALTIME_WEEKLY_FILE_PATH: Optional[Path] = None
    REALTIME_PREDICTIONS_FILE_PATH: Optional[Path] = None

    # CORS origins for frontend web application development
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ]

    @property
    def operational_metrics_file(self) -> Path:
        """Return the path to the operational metrics CSV."""
        if self.METRICS_FILE_PATH is not None:
            return Path(self.METRICS_FILE_PATH)
        return (
            self.PROJECT_ROOT
            / "ml"
            / "realtime"
            / "final_report"
            / "AgriVision_Operational_Metrics.csv"
        )

    @property
    def operational_status_file(self) -> Path:
        """Return the path to the operational status CSV."""
        if self.OPERATIONAL_STATUS_FILE_PATH is not None:
            return Path(self.OPERATIONAL_STATUS_FILE_PATH)
        return (
            self.PROJECT_ROOT
            / "ml"
            / "realtime"
            / "final_report"
            / "AgriVision_Operational_Status_2026.csv"
        )

    @property
    def realtime_weekly_file(self) -> Path:
        """Return the path to the realtime weekly history observations CSV."""
        if self.REALTIME_WEEKLY_FILE_PATH is not None:
            return Path(self.REALTIME_WEEKLY_FILE_PATH)
        return (
            self.PROJECT_ROOT
            / "ml"
            / "realtime"
            / "data"
            / "AgriVision_Realtime_Weekly_Kavathe_Latur.csv"
        )

    @property
    def realtime_predictions_file(self) -> Path:
        """Return the path to the realtime predictions CSV."""
        if self.REALTIME_PREDICTIONS_FILE_PATH is not None:
            return Path(self.REALTIME_PREDICTIONS_FILE_PATH)
        return (
            self.PROJECT_ROOT
            / "ml"
            / "realtime"
            / "predictions"
            / "realtime_predictions_2026.csv"
        )


settings = Settings()
