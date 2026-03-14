import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import yaml
import plotly.graph_objects as go
from prophet.plot import plot_plotly

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="Malaysia Agriculture Analytics",
    layout="wide"
)

# --------------------------------------------------
# Load Data
# --------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("data/final_crop.csv")

@st.cache_data
def load_clusters():
    return pd.read_csv("data/state_clusters.csv")

@st.cache_resource
def load_model_pred():
    return joblib.load("model/best_model.pkl")

@st.cache_resource
def load_model_forecast():
    return joblib.load("model/prophet_crop_models.pkl")

@st.cache_data
def load_config():
    with open("config.yaml","r") as f:
        return yaml.safe_load(f)

config = load_config()

cluster_df = load_clusters()
df = load_data()
df["date"] = pd.to_datetime(df["date"])
prediction_model = load_model_pred()
prophet_models = load_model_forecast()
# -------------------------------------------------
# BACKGROUND FUNCTION
# -------------------------------------------------
def set_background(image_url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{image_url}");
            background-size: cover;
            background-attachment: fixed;
        }}

        header {{visibility: hidden;}}
        footer {{visibility: hidden;}}

        .block-container {{
            padding-top: 0rem !important;
            padding-bottom: 0rem !important;
        }}

        div[role="radiogroup"] > label > div:first-child {{
            display:none !important;
        }}

        div[role="radiogroup"] {{
            gap:30px;
            justify-content:center;
        }}

        div[role="radiogroup"] label {{
            background:none !important;
            border:none !important;
            padding:0 !important;
            cursor:pointer;
        }}

        div[role="radiogroup"] label div {{
            color:rgba(255,255,255,0.7)!important;
            font-size:18px!important;
        }}

        div[role="radiogroup"] label:hover div,
        div[role="radiogroup"] label:has(input:checked) div {{
            color:white!important;
            font-weight:bold!important;
            text-decoration:underline;
            text-underline-offset:8px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


# --------------------------------------------------
# Navigation
# --------------------------------------------------

nav = st.radio(
    "",
    ["Home","Power BI Dashboard","Exploratory Data Analysis", "Cluster Insights", "Prediction", "Forecasting"],
    horizontal=True,
    label_visibility="collapsed"
)

# =================================================
# HOME
# =================================================
if nav == "Home":

    set_background("https://i.pinimg.com/736x/b0/23/3b/b0233bbc37682d2eb28d5692341296b1.jpg")


    # =====================================
    # SECTION 1 : MAIN TITLE (NO IMAGE)
    # =====================================
    st.markdown("""
    <div style="
        padding:120px 20px;
        text-align:center;
    ">

    <h1 style="font-size:90px;color:white;">
    Malaysia Crop Production Analytics
    </h1>

    <h3 style="color:white;">
    Machine Learning Forecast & Prediction
    </h3>

    </div>
    """, unsafe_allow_html=True)



    # =====================================
    # SECTION 2 : ABOUT PROJECT
    # IMAGE + 40% DARK OVERLAY
    # =====================================
    st.markdown("""
    <div style="
        width:100vw;
        margin-left:calc(-50vw + 50%);
        background-color:#062e0c;
        background-size:cover;
        background-position:center;    
    ">

    <div style="
        max-width:1000px;
        margin:auto;
        padding:80px 20px;
        text-align:center;
    ">

    <h2 style="text-align:center;color:white;">
    About This Project
    </h2>

    <br>

    <p style="color:white;font-size:18px;text-align:center;max-width:900px;margin:auto;">
    This project analyzes Malaysia’s crop production patterns across states and years 
    using exploratory data analysis, machine learning, and interactive dashboards.
    </p>

    <br>

    <p style="color:white;font-size:18px;text-align:center;max-width:900px;margin:auto;">
    The goal is to uncover regional crop specialization, production trends, and risk patterns, and translate 
    these insights into decision-ready visuals.
    </p>

    <br>

    <p style="text-align:center;color:white;">
    Data source:
    <a href="https://climateknowledgeportal.worldbank.org/country/malaysia/climate-data-historical" target="_blank" style="color:#FFD700;">Climate Knowledge Portal</a>
    <br>
    <a href="https://open.dosm.gov.my/data-catalogue" target="_blank" style="color:#FFD700;">OpenDOSM</a>
    </p>

    </div>
    """, unsafe_allow_html=True)



    # =====================================
    # SECTION 3 : PROBLEM STATEMENT
    # DARK BACKGROUND
    # =====================================
    st.markdown("""
    <div style="
        width:100vw;
        margin-left:calc(-50vw + 50%);
        background-color:#6e460e;
    ">

    <div style="
        max-width:1000px;
        margin:auto;
        padding:80px 20px;
        text-align:center;
    ">

    <h2 style="text-align:center;color:white;">
    Problem Statement
    </h2>

    <br>

    <p style="color:white;font-size:18px;text-align:center;max-width:900px;margin:auto;">
    Crop production in Malaysia varies across states and over time due to differences in regional specialization, environmental conditions, and agricultural practices. These variations make it difficult for stakeholders to clearly understand production patterns, 
    identify potential risks, and plan for future agricultural output. Although historical agricultural data is available, it is often underutilized for generating actionable insights. This project addresses the gap by analyzing Malaysia’s crop production data to uncover regional trends, identify crop specialization across states,and forecast future production using machine learning and time-series models.
    </p>

    </div>
    """, unsafe_allow_html=True)

# ==================================================
# Power BI dashboard
# ==================================================
elif nav == "Power BI Dashboard":

    set_background("https://64.media.tumblr.com/7e5be0b460f1404bfbf24807efa95f04/5bdfeadfc689526d-6d/s400x600/a87a377cee60d959ae9560c588ec691a2da470db.gif")
    st.title("Interactive Power BI Dashboard")

    power_bi_url = "https://app.powerbi.com/view?r=eyJrIjoiYWYzOTk5NjctNTI4Mi00MGU3LTg1Y2MtYjY3YTg4YzNlNGY0IiwidCI6ImE2M2JiMWE5LTQ4YzItNDQ4Yi04NjkzLTMzMTdiMDBjYTdmYiIsImMiOjEwfQ%3D%3D"

    st.components.v1.iframe(
        power_bi_url,
        width=1200,
        height=800
    )

# ==================================================
# PAGE 2 – EDA
# ==================================================
elif nav == "Exploratory Data Analysis":

    set_background("https://64.media.tumblr.com/7e5be0b460f1404bfbf24807efa95f04/5bdfeadfc689526d-6d/s400x600/a87a377cee60d959ae9560c588ec691a2da470db.gif")
    st.title("Production Trend Analysis")

    crop_selection = st.selectbox(
        "Select Crop Type",
        sorted(df["crop_type"].unique())
    )

    filtered = df[df["crop_type"] == crop_selection]

    # Use log scale for visualization (because skewed)
    filtered["log_production"] = np.log10(filtered["production"] + 1)

    fig = px.line(
        filtered,
        x="year",
        y="log_production",
        color="state",
        title=f"{crop_selection} Production Trend (Log Scale)"
    )

    st.plotly_chart(fig, use_container_width=True)

    # Distribution
    st.subheader("Distribution of Total Production by State")

    state_total = df.groupby("state")["production"].sum().reset_index()
    state_total["log_total"] = np.log10(state_total["production"] + 1)

    fig2 = px.histogram(
        state_total,
        x="log_total",
        nbins=10,
        title="Distribution of Total Production (Log Scale)"
    )

    st.plotly_chart(fig2, use_container_width=True)

    st.caption("Log scale used due to high production skewness (billions vs thousands).")

# ==================================================
# PAGE 3 – CLUSTERING
# ==================================================
elif nav == "Cluster Insights":

    st.title("State Clustering by Agricultural Profile")

    fig = px.scatter(
        cluster_df,
        x="avg_production",
        y="avg_growth_rate",
        color="cluster_label",
        size="variability",
        hover_name="state",
        title="Clustered States by Production & Growth"
    )

    # Apply log scale ONLY to axis display
    fig.update_xaxes(type="log")

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    ### Cluster Interpretation
    - **Cluster 0**: Flower Specialized  
    - **Cluster 1**: Mixed Crop  
    - **Cluster 2**: Flower + Rice focused
    """)

# ==================================================
# PAGE 4 – ML PREDICTION
# ==================================================
elif nav == "Prediction":

    set_background("https://64.media.tumblr.com/7e5be0b460f1404bfbf24807efa95f04/5bdfeadfc689526d-6d/s400x600/a87a377cee60d959ae9560c588ec691a2da470db.gif")

    st.title("📊 Crop Production Prediction")

    st.markdown("""
    Model trained using log-transformed production values  
    Final model selected based on lowest RMSE and highest R².
    """)

    # -------------------------------------------------
    # CUSTOM CONTAINER STYLING
    # -------------------------------------------------
    st.markdown("""
    <style>

    /* -------- GLOBAL TEXT -------- */
    html, body, [class*="css"]  {
        font-family: 'Segoe UI', sans-serif;
    }

    /* Hover animation */
    [class*="st-key-col_"]:hover{
        transform: translateY(-6px);
        box-shadow:
            0px 10px 35px rgba(0,0,0,0.9),
            0px 0px 10px rgba(255,255,255,0.15);
    }

    /* -------- HEADER STYLE -------- */
    h1, h2, h3 {
        color: white;
        font-weight: 600;
    }

    /* -------- INPUT BOXES -------- */
    .stTextInput input,
    .stNumberInput input,
    .stSelectbox div,
    .stSlider {

        background-color: rgba(255,255,255,0.08) !important;
        color: white !important;

        border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.15);

    }

    /* -------- BUTTON STYLE -------- */
    .stButton>button {

        background: linear-gradient(135deg,#00c6ff,#0072ff);
        color: white;
        border-radius: 12px;

        border: none;
        padding: 10px 20px;

        font-weight: 600;

        box-shadow: 0px 4px 12px rgba(0,0,0,0.5);

        transition: all 0.2s ease;

    }

    /* Button hover */
    .stButton>button:hover {

        transform: scale(1.05);
        box-shadow: 0px 6px 18px rgba(0,0,0,0.7);

    }

    /* -------- SUCCESS BOX -------- */
    .stSuccess {

        border-radius: 10px;
        background: rgba(46, 204, 113,0.2);
        border: 1px solid rgba(46,204,113,0.5);

    }

    /* -------- SLIDER COLOR -------- */
    .stSlider div[data-baseweb="slider"] > div {
        background-color: #00c6ff;
    }

    </style>
    """, unsafe_allow_html=True)

        # -------------------------------------------------
        # COLUMNS
        # -------------------------------------------------
    col1, col2, col3 = st.columns(3)

        # -------------------------------------------------
        # MODEL INFO
        # -------------------------------------------------
    with col1:
        with st.container(key="col_info"):

            st.subheader("ℹ️ Prediction Model Info")

            st.write("Algorithm:", config["model"]["algorithm_pred"])
            st.write("R square score:", config["model"]["r_square_score"])
            st.write("RMSE:", config["model"]["rmse"])
            st.write("MAE:", config["model"]["MAE"])

    # -------------------------------------------------
    # PREDICTION INPUT
    # -------------------------------------------------
    with col2:
        with st.container(key="col_tool"):

            st.subheader("✨ Prediction Tool")

            crop = st.selectbox("Crop Type", sorted(df["crop_type"].unique()))
            state = st.selectbox("State", sorted(df["state"].unique()))

            planted_area_input = st.text_input(
                "Planted Area (hectares)",
                "1000"
            )

            precipitation_input = st.text_input(
                "Precipitation (mm)",
                "200"
            )

    # -------------------------------------------------
    # YEAR + PREDICT BUTTON
    # -------------------------------------------------
    with col3:
        with st.container(key="col_year"):

            year = st.slider("Year", int(df["year"].min()), int(df["year"].max()+3), 2023)

            if st.button("Predict Production"):

                try:
                    planted_area = float(planted_area_input)
                    precipitation = float(precipitation_input)

                    input_df = pd.DataFrame({
                        "state": [state],
                        "crop_type": [crop],
                        "planted_area": [planted_area],
                        "precipitation": [precipitation],
                        "year": [year]
                    })

                    log_prediction = prediction_model.predict(input_df)

                    prediction = np.expm1(log_prediction[0])

                    st.success(f"Predicted Production: {prediction:,.2f}")

                    st.caption(
                        "Prediction converted back from log scale to actual production value."
                    )

                except ValueError:
                    st.error("Please enter valid numeric values for Planted Area and Precipitation.")
# =================================================
# ML FORECASTING
# =================================================
elif nav == "Forecasting":

    set_background("https://64.media.tumblr.com/7e5be0b460f1404bfbf24807efa95f04/5bdfeadfc689526d-6d/s400x600/a87a377cee60d959ae9560c588ec691a2da470db.gif")

    st.title("📈 Crop Production Forecast")

    col1,col2 = st.columns(2)

    with col1:
        
        st.subheader("ℹ️ Forecasting Model Info")
        st.write("Algorithm:",config["model"]["algorithm_forecast"])
        st.write("RMSE:",config["model"]["rmse_forecast"])
        st.write("MAPE:",config["model"]["MAPE"])
        
        st.image(
            "https://i.pinimg.com/originals/7c/6e/ea/7c6eeaeb617ad2c17d567c7ff9621e17.gif",
            width=500
        )

    with col2:

        st.subheader("✨ Forecast Tool")
    # --------------------------------------
    # Crop Selection
    # --------------------------------------

        crop_list = list(prophet_models.keys())

        selected_crop = st.selectbox(
            "Select Crop for Forecast",
            crop_list
        )

        model = prophet_models[selected_crop]

    # --------------------------------------
    # Forecast Horizon
    # --------------------------------------

        forecast_years = st.slider(
            "Forecast Years",
            1,
            10,
            3
        )

        periods = forecast_years * 12

        future = model.make_future_dataframe(
            periods=periods,
            freq="M"
        )

        forecast = model.predict(future)

    # --------------------------------------
    # Interactive Plot
    # --------------------------------------

        st.subheader("Production Forecast")

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=forecast["ds"],
            y=forecast["yhat"],
            name="Forecast",
            line=dict(width=3)
        ))

        fig.add_trace(go.Scatter(
            x=forecast["ds"],
            y=forecast["yhat_upper"],
            name="Upper Confidence",
            line=dict(width=0, dash="dot")
        ))

        fig.add_trace(go.Scatter(
            x=forecast["ds"],
            y=forecast["yhat_lower"],
            fill='tonexty',
            fillcolor='rgba(0,100,80,0.2)',
            name="Confidence Interval",
            line=dict(width=0, dash="dot")
        ))

        fig.update_layout(
            title=f"{selected_crop} Production Forecast",
            xaxis_title="Year",
            yaxis_title="Production",
            template="plotly_white"
        )

        st.plotly_chart(fig, use_container_width=True)

    # --------------------------------------
    # Components Plot
    # --------------------------------------

        st.subheader("Trend & Seasonality")

        comp_fig = model.plot_components(forecast)

        st.pyplot(comp_fig)
# -------------------------------------------------
# FOOTER
# -------------------------------------------------
footer_html = """
<style>

.footer {
position: relative;
left: 0;
bottom: 0;
width: 100%;
background-color: transparent;
color: white;
text-align: center;
padding: 20px 0;
font-size: 14px;
z-index: 9999;
}

.footer a {
margin: 0 10px;
text-decoration: none;
}

.footer img {
width: 28px;
margin-left: 8px;
margin-right: 8px;
vertical-align: middle;
transition: transform 0.2s;
}

.footer img:hover {
transform: scale(1.2);
}

</style>

<div class="footer">

<p>
Built with Data & Passion | © 2026 Jenifer M Jues
</p>

<a href="https://github.com/JMJ-ai/Exploratory-Crop-Production-Analytic-" target="_blank">
<img src="https://cdn-icons-png.flaticon.com/512/25/25231.png">
</a>

<a href="https://www.linkedin.com/in/jenifermayangjues/" target="_blank">
<img src="https://cdn-icons-png.flaticon.com/512/174/174857.png">
</a>

<a href="https://icons8.com/" target="_blank">
<img src="https://img.icons8.com/?size=100&id=ayJDJ6xQKgM6&format=png&color=000000">
</a>

<a href="mailto:jeniferjues@gmail.com">
<img src="https://cdn-icons-png.flaticon.com/512/732/732200.png">
</a>

</div>
"""

st.markdown(footer_html, unsafe_allow_html=True)
