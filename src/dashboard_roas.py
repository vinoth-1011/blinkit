import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
import plotly.express as px

# ML imports
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# ---------- DB CONFIG ----------
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "postgres"
DB_USER = "postgres"
DB_PASSWORD = "1234"

engine = create_engine(
    f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")


@st.cache_data
def load_roas_data():
    query = "SELECT * FROM master_daily_roas ORDER BY date;"
    df = pd.read_sql(query, engine)
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_resource
def load_ml_model():
    # Train the model fresh from the CSV on app start
    feedback_path = r"D:\vs_code\GUVI\blinkit_project\data_raw\Blinkit - blinkit_customer_feedback.csv"
    fb = pd.read_csv(feedback_path)

    # 1 = Positive sentiment, 0 = Neutral/Negative
    fb["is_positive"] = (fb["sentiment"] == "Positive").astype(int)

    data = fb[["feedback_text", "is_positive"]].dropna()

    X = data["feedback_text"]
    y = data["is_positive"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words="english"
        )),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    model.fit(X_train, y_train)
    return model


# ---------- UI LAYOUT ----------
st.set_page_config(page_title="Blinkit Analytics", layout="wide")

st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio(
    "Go to", ["ROAS Dashboard", "Feedback Sentiment Predictor"])

# ---------- PAGE 1: ROAS DASHBOARD ----------
if page == "ROAS Dashboard":
    df = load_roas_data()

    st.title("📊 Blinkit ROAS Dashboard")

    total_spend = df["total_spend"].sum()
    total_revenue = df["total_revenue"].sum()
    avg_roas = (total_revenue / total_spend) if total_spend > 0 else 0

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Spend", f"₹{total_spend:,.0f}")
    c2.metric("Total Revenue", f"₹{total_revenue:,.0f}")
    c3.metric("Avg ROAS", f"{avg_roas:.2f}x")

    # Date filter
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        start_date = st.date_input("Start Date", df["date"].min())
    with col_f2:
        end_date = st.date_input("End Date", df["date"].max())

    mask = (df["date"] >= pd.to_datetime(start_date)) & (
        df["date"] <= pd.to_datetime(end_date))
    filtered = df[mask]

    # Spend vs Revenue
    fig = px.line(filtered, x="date", y=["total_spend", "total_revenue"],
                  title="Daily Spend & Revenue")
    st.plotly_chart(fig, use_container_width=True)

    # ROAS label as text preview
    st.subheader("Sample Data")
    st.dataframe(filtered.head())

# ---------- PAGE 2: FEEDBACK SENTIMENT PREDICTOR ----------
# ---------- PAGE 2: FEEDBACK SENTIMENT PREDICTOR ----------
# ---------- PAGE 2: FEEDBACK SENTIMENT PREDICTOR ----------
else:
    st.title("💬 Feedback Sentiment Predictor")

    model = load_ml_model()

    st.write("Enter a customer feedback text and the model will predict whether it is **Positive** or **Not Positive**.")

    user_text = st.text_area("Feedback Text", height=150,
                             placeholder="Example: 'Delivery was quick and the products were fresh!'")

    if st.button("Predict Sentiment"):
        if not user_text.strip():
            st.warning("Please enter some feedback text.")
        else:
            # Always predict sentiment first
            pred = model.predict([user_text])[0]

            # Default: No confidence
            confidence = None

            # Try to get predict_proba safely
            try:
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba([user_text])[0]
                    confidence = float(max(proba)) * 100
            except Exception:
                confidence = None

            label = "Positive 😊" if pred == 1 else "Not Positive 😐"

            st.subheader(f"Prediction: {label}")

            if confidence is not None:
                st.write(f"Confidence: **{confidence:.1f}%**")
            else:
                st.caption(
                    "Confidence is not available in this model configuration.")
