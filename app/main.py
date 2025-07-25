from fastapi import FastAPI
from app.routes import hackrx

app = FastAPI()
app.include_router(hackrx.router)
