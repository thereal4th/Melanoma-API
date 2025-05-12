# main.py
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from app.inference import load_models, predict_melanoma
from PIL import Image
import io

app = FastAPI()

# Load models at startup
seg_model, clf_model = load_models()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        prob, _, _, _ = predict_melanoma(image, seg_model, clf_model) # only take probabilities and skip other returned values for now

        return JSONResponse(content={"probability": round(prob, 4)})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
