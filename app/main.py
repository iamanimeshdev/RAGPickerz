from fastapi import FastAPI
from app.routers.hackrx import router as hackrx_router

app = FastAPI()

app.include_router(hackrx_router)