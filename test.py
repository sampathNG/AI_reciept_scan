from fastapi import FastAPI
import uvicorn  
from .controllers import router
app = FastAPI()

# Include the router from routes.py
app.include_router(router)

if __name__ == "__main__":
    # Additional configuration (optional)
    # For example, setting environment variables for database connection

    # Run the uvicorn server
    uvicorn.run(app, host="0.0.0.0", port=8000) 
