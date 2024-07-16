# Imports
import joblib
import warnings
from fastapi import FastAPI
from star_properties import StarInput
from star_types_data import star_types

# Ignore the warnings due to version changes
warnings.filterwarnings('ignore')

# Create the fastAPI Instance (fastapi app)
app = FastAPI()

# Load the pipeline
path_pipeline = 'startype_pipeline.joblib'
pipeline = joblib.load(path_pipeline)

@app.get("/")
def read_root():
    return {"message" : "App Running!"}

# Get the predictions using post request
@app.post("/predict")
def prediction(input_data : StarInput):
    # Get the input data
    data = [[
        input_data.temperature,
        input_data.luminosity,
        input_data.radius,
        input_data.absolute_magnitude
    ]]

    # Make the predictions
    prediction = pipeline.predict(data)[0]
    confidence_score = pipeline.predict_proba(data)[0][prediction]
    print(pipeline.predict_proba(data)[0])
    return {
        'Predicted Star Type' : star_types[prediction],
        'Confidence Score' : str(round(confidence_score*100, 1)) + "%"
        }

