import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

ALLOWED_EXTENSIONS = {'jpeg', 'jpg', 'png'}

DEBUG = True
SECRET_KEY = os.getenv('SECRET_KEY', 'default_fallback_key_if_not_set')
