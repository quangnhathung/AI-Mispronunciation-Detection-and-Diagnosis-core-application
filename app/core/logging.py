import sys
from loguru import logger
from app.core.config import settings


def setup_logging() -> None:
    logger.remove()
    logger.add(
        sys.stdout,
        format=settings.log_format,
        level=settings.log_level,
        colorize=True,
    )
    log_file = settings.log_dir / "app.log"
    logger.add(
        str(log_file),
        format=settings.log_format,
        level=settings.log_level,
        rotation="10 MB",
        retention="30 days",
        compression="gz",
    )
    logger.debug(f"Logging initialized at {settings.log_level} level")
