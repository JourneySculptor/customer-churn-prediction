from uvicorn import run
import os
from api import app  # Import the FastAPI app instance from api.py

if __name__ == "__main__":
    # Run the FastAPI app using Uvicorn as the ASGI server
    port = int(os.environ.get("PORT", 8000))   # Run on 0.0.0.0 to make it accessible externally
    run("main:app", host="0.0.0.0", port=port)