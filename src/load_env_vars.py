import os
import sys

from dotenv import load_dotenv

load_dotenv()
OUTPUT_DIRECTORY = os.getenv("DOCS_PATH")
SEARCH_DIRECTORY = os.getenv("JSON_PATH")
BING_SUB_KEY = os.getenv("BING_SUBSCRIPTION_KEY")
BING_SEARCH_KEY = os.getenv("BING_SEARCH_URL")

if OUTPUT_DIRECTORY and SEARCH_DIRECTORY:
    print("Exported environment variables successfully!")
    if not os.path.exists(OUTPUT_DIRECTORY):
        os.makedirs(OUTPUT_DIRECTORY)
    if not os.path.exists(SEARCH_DIRECTORY):
        os.makedirs(SEARCH_DIRECTORY)
else:
    print(
        'Please set the environment variables using the keys "DOCS_PATH", "JSON_PATH" in the .env file in the root directory!'
    )
    sys.exit(0)
