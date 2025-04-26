import logging

logging.basicConfig(
    filename="log/app.log",
    encoding="utf-8",
    filemode="a",
    level=logging.INFO,
    format="%(levelname)s: %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M"
)

logger = logging.getLogger(__name__)
