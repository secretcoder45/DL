# 🎉 Dynamic Venue Pricing with FastAPI & Machine Learning

## 📌 Project Overview

This project is a **machine learning--powered web application** that
predicts **dynamic prices for event venues** based on:\
- 📍 Location\
- 👥 Capacity\
- 📅 Event Date

It integrates a **LightGBM regression model** with a **FastAPI backend**
and a **simple HTML/Jinja2 frontend**, allowing users to input event
details and receive real-time dynamic price predictions.

------------------------------------------------------------------------

## 🚀 Features

-   ✅ Machine learning model (LightGBM) trained on historical booking
    data.\
-   ✅ REST API built with **FastAPI**.\
-   ✅ Clean and simple **HTML frontend** using **Jinja2 templates**.\
-   ✅ Predicts **venue prices dynamically** based on event details.\
-   ✅ Outputs predictions in a **tabular format** for easy
    interpretation.

------------------------------------------------------------------------

## 🛠️ Tech Stack

-   **Backend**: FastAPI (Python)\
-   **Frontend**: HTML + Jinja2 Templates\
-   **Machine Learning**: LightGBM Regressor\
-   **Data Handling**: Pandas, Joblib\
-   **Server**: Uvicorn

------------------------------------------------------------------------

## 📂 Project Structure

    project-root/
    │── backend/
    │   ├── main.py              # FastAPI app (ML model + routes)
    │   ├── model.pkl            # Trained LightGBM model
    │   ├── le_location.pkl      # Label encoder for location
    │   ├── eng_df.csv           # Dataset with features
    │
    │── frontend/
    │   ├── website.html         # Input form (Location, Capacity, Date)
    │   ├── results.html         # Tabular display of predictions
    │
    │── README.md                # Project documentation

------------------------------------------------------------------------

## ⚙️ Installation & Setup

1.  **Clone the repository**\

``` bash
git clone https://github.com/your-username/dynamic-venue-pricing.git
cd dynamic-venue-pricing
```

2.  **Install dependencies**\

``` bash
pip install fastapi uvicorn pandas joblib lightgbm jinja2
```

3.  **Run the FastAPI server**\

``` bash
uvicorn backend.main:app --reload
```

4.  **Access the app in your browser**\

```{=html}
<!-- -->
```
    http://127.0.0.1:8000

------------------------------------------------------------------------

## 🎯 Usage

1.  Open the **form page** at `http://127.0.0.1:8000`.\
2.  Select a **location**, enter **capacity**, and choose an **event
    date**.\
3.  Click **Get Price**.\
4.  The app will display **predicted venue prices** in a **table
    format**.

------------------------------------------------------------------------

## 📸 Screenshots

### 🖊️ Input Form

*User selects location, enters capacity & event date.*\
![Form Screenshot](frontend/form.png)

### 📊 Prediction Results

*Output in a clean tabular format.*\
![Results Screenshot](frontend/results.png)

------------------------------------------------------------------------

## 👨‍💻 Contributors

-   **Your Name** -- Developer / ML Engineer

------------------------------------------------------------------------

## 📜 License

This project is licensed under the **MIT License** -- feel free to use
and modify.
