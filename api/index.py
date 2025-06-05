from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()

@app.post("/")
async def answer(request: Request):
    body = await request.json()
    question = body.get("question", "")
    # For now, dummy reply
    return JSONResponse({
        "answer": f"You asked: {question}",
        "links": []
    })
