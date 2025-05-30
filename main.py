# main.py
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from app.inference import load_models, predict_melanoma, predict_melanoma_nosegment
from PIL import Image
import io
import traceback

app = FastAPI()

# Load models at startup
seg_model, clf_model = load_models()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or replace "*" with specific mobile app domains for better security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

'''@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        prob, _, _, _ = predict_melanoma(image, seg_model, clf_model) # only take probabilities and skip other returned values for now

        return JSONResponse(content={"probability": round(prob, 4)})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})'''

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:

        print("Received a request to /predict", flush=True)

        # TEMP Load models inside the route for testing
        # seg_model, clf_model = load_models()

        contents = await file.read()
        print(f"File size: {len(contents)} bytes", flush=True)

        image = Image.open(io.BytesIO(contents)).convert("RGB")
        print("Image loaded successfully", flush=True)

        prob, _, _, _ = predict_melanoma(image, seg_model, clf_model)
        print(f"Prediction probability: {prob}", flush=True)

        return JSONResponse(content={"probability": round(prob, 4)})

    except Exception as e:
        print(f"Exception in /predict: {e}", flush=True)
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/predict_nosegment")
async def predict_nosegment(file: UploadFile = File(...)):
    try:
        print("Received a request to /predict_nosegment")

        contents = await file.read()

        image = Image.open(io.BytesIO(contents)).convert("RGB")

        prob, _, _ = predict_melanoma_nosegment(image, clf_model)
        print(f"Prediction probability: {prob}", flush=True)

        return JSONResponse(content={"probability": round(prob, 4)})

    except Exception as e:
        print(f"Exception in /predict: {e}", flush=True)
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/")
def read_root():
    print("get request received", flush = True)
    return {"message": "It works!"}

@app.post("/test-upload")
async def test_upload(file: UploadFile = File(...)):
    contents = await file.read()
    print(f"🧪 test-upload: Received {file.filename} of size {len(contents)} bytes")
    print("running test-upload", flush = True)
    return {"filename": file.filename, "size_kb": len(contents) // 1024}
    
    #seg_model, clf_model = load_models()



