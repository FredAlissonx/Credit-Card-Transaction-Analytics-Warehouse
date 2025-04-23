from dotenv import load_dotenv
from log.setup import logger
import requests

load_dotenv()

session = requests.session()

logger = logger