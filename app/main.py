from fastapi import FastAPI
from app.routers import upload, query

app = FastAPI()

app.include_router(upload.router)
app.include_router(query.router)
