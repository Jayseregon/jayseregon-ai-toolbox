from fastapi import FastAPI

from src.routes.embedding import router as embedding_router

app = FastAPI()


@app.get("/")
async def read_root():
    return {"greetings": "Welcome to Jayseregon AI toolbox API."}


app.include_router(embedding_router)
