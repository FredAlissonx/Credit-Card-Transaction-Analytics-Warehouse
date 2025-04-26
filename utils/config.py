from dotenv import load_dotenv
from log.setup import logger
from datetime import datetime
import requests

load_dotenv()

session = requests.session()

logger = logger

current_date = datetime.now()