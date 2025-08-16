from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
import joblib
import os

app = FastAPI()
templates = Jinja2Templates(directory="../frontend")

# ---------- Load model & encoder at startup ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load trained model
model_path = os.path.join(BASE_DIR, "model.pkl")
model = joblib.load(model_path)

# Load label encoder
encoder_path = os.path.join(BASE_DIR, "le_location.pkl")
le_location = joblib.load(encoder_path)

# Load original dataset
data_path = os.path.join(BASE_DIR, "eng_df.csv")
original_df = pd.read_csv(data_path)


# ---------- API Endpoints ----------
@app.get("/", response_class=HTMLResponse)
async def get_form(request: Request):
    return templates.TemplateResponse("website.html", {"request": request})

@app.post("/submit", response_class=HTMLResponse)
async def predict_price(
    request: Request,
    location: str = Form(...),
    capacity: int = Form(...),
    date: str = Form(...)
):
    print(f"Location: {location}")
    print(f"Capacity: {capacity}")
    print(f"Date: {date}")

    # Convert date
    event_date = pd.to_datetime(date)

    # Filter relevant venues
    venue_subset = original_df[
        (original_df['location'].str.lower() == location.lower()) &
        (original_df['capacity'].between(capacity * 1, capacity * 1.5))
    ].copy()

    if venue_subset.empty:
        return templates.TemplateResponse(
            "results.html",
            {
                "request": request,
                "message": "No venues found for selected filters.",
                "results": []
            }
        )

    # Feature engineering
    venue_subset['event_date'] = pd.to_datetime(date)
    venue_subset['lead_time_days'] = (event_date - pd.to_datetime("2024-12-01")).days
    venue_subset['is_weekend'] = event_date.weekday() in [5, 6]

    # Encode location
    venue_subset['location'] = le_location.transform(venue_subset['location'])

    # Ensure categorical dtypes match training
    categorical_cols = ['event_date', 'season', 'lead_time_bucket', 'location']
    for col in categorical_cols:
        if col in venue_subset.columns:
            venue_subset[col] = venue_subset[col].astype('category')
    
    # Features expected by model
    feature_cols = [
        'venue_id','event_date', 'lead_time_days', 'season', 'is_weekend', 'special_event', 'capacity','latitude', 'longitude',
        'geo_cluster', 'avg_price_last_14_days', 'bookings_last_14_days',
        'venue_avg_price_overall', 'lead_time_bucket', 'location'
    ]
    venue_subset = venue_subset.groupby(['venue_id', 'location'], as_index=False).agg({
        'event_date': 'first',             
        'lead_time_days': 'first',         
        'season': 'first',                 
        'is_weekend': 'first',             
        'special_event': 'max',            
        'capacity': 'mean',                
        'latitude': 'mean',                
        'longitude': 'mean',               
        'geo_cluster': 'first',            
        'avg_price_last_14_days': 'mean',  
        'bookings_last_14_days': 'mean',  
        'venue_avg_price_overall': 'mean', 
        'lead_time_bucket': 'first'        
    })

    # Predict
    preds = model.predict(venue_subset[feature_cols]) 

    # Keep venue_id and location for output
    output_cols = venue_subset[['venue_id', 'location', 'capacity']].copy()

    # Add predictions to output
    output_cols['predicted_price'] = preds.round(0) 
    results = output_cols.to_dict(orient='records')  

    # Return in HTML file
    return templates.TemplateResponse(
        "prediction.html",
        {
            "request": request,
            "results": results,
            "location": location,
            "capacity": capacity,
            "date": date
        }
    )