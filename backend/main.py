from fastapi import FastAPI, Form
import pandas as pd
import joblib

app = FastAPI()

model = joblib.load("lightgbm_model.pkl")
original_df = pd.read_csv("eng_df.csv")

@app.post("/submit")
async def submit_data(
    location: str = Form(...),
    capacity: int = Form(...),
    date: str = Form(...)
):
    event_date = pd.to_datetime(date)

    venue_subset = original_df[
        (original_df['location'].str.lower() == location.lower()) &
        (original_df['capacity'].between(capacity * 0.7, capacity * 1.5))
    ].copy()

    if venue_subset.empty:
        return {"message": "No venues found for selected filters."}

    venue_subset['event_date'] = event_date
    venue_subset['lead_time_days'] = (event_date - pd.to_datetime("2024-12-01")).days
    # venue_subset['day_of_week'] = event_date.weekday()
    # venue_subset['is_weekend'] = venue_subset['day_of_week'].isin([5, 6]).astype(int)

    feature_cols = [
        'capacity', 'lead_time_days', 'is_weekend',
        'avg_price_last_14_days', 'bookings_last_14_days', 'venue_avg_price_overall',
        'lead_time_bucket', 'location'
    ]

    venue_subset['predicted_price'] = model.predict(venue_subset[feature_cols])

    return {
        "results": venue_subset[['venue_id', 'location', 'predicted_price']].to_dict(orient='records')
    }