from fastapi import FastAPI, Form
import pandas as pd
import joblib
import os

app = FastAPI()

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
@app.post("/submit")
async def predict_price(
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
        (original_df['capacity'].between(capacity * 0.9, capacity * 1.2))
    ].copy()

    if venue_subset.empty:
        return {"message": "No venues found for selected filters."}

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
    
    # print(venue_subset.dtypes)

    # Features expected by model
    feature_cols = [
        'venue_id','event_date', 'lead_time_days', 'season', 'is_weekend', 'special_event', 'capacity','latitude', 'longitude',
        'geo_cluster', 'avg_price_last_14_days', 'bookings_last_14_days',
        'venue_avg_price_overall', 'lead_time_bucket', 'location'
    ]
    # Predict
    preds = model.predict(venue_subset[feature_cols])


    # Keep venue_id and location for output
    output_cols = venue_subset[['venue_id', 'location']].copy()

    # Add predictions to output
    output_cols['predicted_price'] = preds

    return {
        "results": output_cols.to_dict(orient='records')
    }


