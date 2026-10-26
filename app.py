import streamlit as st                      # streamlit: turns Python code into a web page
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Personality Predictor", page_icon="🧠")


# ---------- 1. Load data, clean it, train the model ----------
@st.cache_resource                           # run only once, not on every click
def train_model():
    df = pd.read_csv("personality_dataset.csv")

    num_cols = ["Time_spent_Alone", "Social_event_attendance", "Going_outside",
                "Friends_circle_size", "Post_frequency"]
    cat_cols = ["Stage_fear", "Drained_after_socializing"]

    for col in num_cols:                     # fill empty cells (same as in the notebook)
        df[col] = df[col].fillna(df[col].median())
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    df["Stage_fear"] = df["Stage_fear"].map({"Yes": 1, "No": 0})
    df["Drained_after_socializing"] = df["Drained_after_socializing"].map({"Yes": 1, "No": 0})
    df["Personality"] = df["Personality"].map({"Extrovert": 1, "Introvert": 0})

    X = df.drop("Personality", axis=1)
    y = df["Personality"]

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)                          # train on all data
    return model, list(X.columns)


model, columns = train_model()

# ---------- 2. Page design ----------
st.title("🧠 Personality Prediction")
st.write("Answer the questions below and the model will predict whether you are an **Introvert** or an **Extrovert**.")

# ---------- 3. Input widgets (one per survey question) ----------
time_alone = st.slider("Hours spent alone per day", 0, 11, 4)
stage_fear = st.radio("Do you have stage fear?", ["No", "Yes"], horizontal=True)
events = st.slider("Social events attended (per month)", 0, 10, 4)
outside = st.slider("How often do you go outside? (0-7)", 0, 7, 4)
drained = st.radio("Do you feel drained after socializing?", ["No", "Yes"], horizontal=True)
friends = st.slider("Size of your close friends circle", 0, 15, 5)
posts = st.slider("Social media posts per week", 0, 10, 4)

# ---------- 4. Predict when the button is clicked ----------
if st.button("Predict my personality"):
    person = pd.DataFrame([[
        time_alone,
        1 if stage_fear == "Yes" else 0,
        events,
        outside,
        1 if drained == "Yes" else 0,
        friends,
        posts,
    ]], columns=columns)

    result = model.predict(person)[0]
    confidence = model.predict_proba(person)[0].max() * 100

    if result == 1:
        st.success(f"Prediction: **Extrovert** ({confidence:.1f}% confident)")
    else:
        st.info(f"Prediction: **Introvert** ({confidence:.1f}% confident)")

st.caption("Made for AI Application Lab, Section 214B. This is a simple ML model, not a real psychological test.")
