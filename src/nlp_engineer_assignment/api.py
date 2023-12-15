from fastapi import FastAPI
from starlette.responses import RedirectResponse
import torch
from nlp_engineer_assignment import model_predict
import main
from pydantic import BaseModel

model = torch.load("/Users/daguila/nlp-engineer-assignment/model_97.8.pth")


app = FastAPI(
    title="NLP Engineer Assignment",
    version="1.0.0"
)

@app.get("/", include_in_schema=False)
async def index():
    """
    Redirects to the OpenAPI Swagger UI
    """
    return RedirectResponse(url="/docs")

class Prediction(BaseModel):
    text: str

@app.post("/predictText")
async def getPredictionFromText(request: Prediction):
    requestlist = []
    requestlist.append(request.text)
    #preprocess is expecting a list of requests
    preprocessed_text = main.preprocess(requestlist)

    prediction = model_predict(model,preprocessed_text)
    #since model_predict returns a list of predictions, if we only have one prediction, we need to get the first item of the list
    return {"prediction": ''.join(str(num) for num in prediction[0].tolist())}