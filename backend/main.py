from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/submit")
async def submit_data(
    location: str = Form(...),
    capacity: int = Form(...),
    date: str = Form(...)
):
    print("📩 Received Booking:")
    print("Location:", location)
    print("Capacity:", capacity)
    print("Date:", date)
    return {"message": "Booking data received successfully!"}