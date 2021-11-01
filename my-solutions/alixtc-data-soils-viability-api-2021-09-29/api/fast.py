

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)




@app.get("/")
async def root():
    return {"Status": "Up and running"}

@app.get("/predict")
def prediction(measure_index, measure_moisture, 
               measure_temperature, measure_chemicals, measure_biodiversity, 
               main_element, soil_condition,
               datetime_start, datetime_end):
    
        parameters = {
            'measure_index': float(measure_index),
            'measure_moisture': float(measure_moisture),
            'measure_temperature': float(measure_temperature),
            'measure_chemicals': float(measure_chemicals),
            'measure_biodiversity': float(measure_biodiversity),
            'main_element': str(main_element),
            'soil_condition': str(soil_condition),
            'datetime_start': str(datetime_start),
            'datetime_end': str(datetime_end)}


        data = pd.DataFrame.from_dict(parameters,orient='index')
        data = data.T
        model = joblib.load('assets/model.joblib')
        results = model.predict(data)[0]
        results = (results == 1) # Transform value to bool to avoid bool as existence test
        return {'viable':bool(results)}
