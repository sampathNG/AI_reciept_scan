from fastapi import FastAPI, Depends
app = FastAPI()
from controllers.ocr import router as router
app.include_router(router)