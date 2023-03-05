import uvicorn
from pydantic import BaseModel
from fastapi import FastAPI, Depends
from ML.predmodel import Model, get_model


class SentimentRequest(BaseModel):
    text: str


class SentimentResponse(BaseModel):
    sentiment: str


app = FastAPI()


@app.post("/predict", response_model=SentimentResponse)
def predict(input: SentimentRequest, model: Model = Depends(get_model)):
    prediction = model.predict(input.text)
    return SentimentResponse(sentiment=prediction)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
