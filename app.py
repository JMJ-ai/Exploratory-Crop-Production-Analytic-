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

    st.markdown("""
    <style>

    /* White glass container */
    [class*="st-key-eda_"], 
    [class*="st-key-cluster_"] {

        background: rgba(255,255,255,0.40);
        backdrop-filter: blur(8px);

        padding: 25px;
        border-radius: 16px;

        border: 1px solid rgba(255,255,255,0.6);

        box-shadow: 
            0px 6px 25px rgba(0,0,0,0.35);

        transition: all 0.25s ease;
    }

    /* Hover animation */
    [class*="st-key-eda_"]:hover,
    [class*="st-key-cluster_"]:hover {

        transform: translateY(-4px);
        box-shadow: 
            0px 10px 30px rgba(0,0,0,0.45);

    }

    </style>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    # ----------------------------------
    # Production Trend Chart
    # ----------------------------------

    with col1:
        with st.container(key="eda_trend"):

            st.subheader("Production Trend by State")

            crop_selection = st.selectbox(
                "Select Crop Type",
                sorted(df["crop_type"].unique())
            )

            filtered = df[df["crop_type"] == crop_selection].copy()

            filtered["log_production"] = np.log10(filtered["production"] + 1)

            fig = px.line(
                filtered,
                x="year",
                y="log_production",
                color="state",
                title=f"{crop_selection} Production Trend (Log Scale)"
            )

            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)"
            )

            st.plotly_chart(fig, use_container_width=True)

    # ----------------------------------
    # Distribution Chart
    # ----------------------------------

    with col2:
        with st.container(key="eda_distribution"):

            st.subheader("Distribution of Total Production by State")

            state_total = df.groupby("state")["production"].sum().reset_index()

            state_total["log_total"] = np.log10(state_total["production"] + 1)

            fig2 = px.histogram(
                state_total,
                x="log_total",
                color="state",
                nbins=10,
                title="Distribution of Total Production (Log Scale)"
            )

            fig2.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)"
            )

            st.plotly_chart(fig2, use_container_width=True)

            st.caption("Log scale used due to high production skewness.")

    # ----------------------------------
    # Heatmap
    # ----------------------------------

    with st.container(key="eda_heatmap"):

        st.subheader("Crop Production Heatmap")

        df_heatmap = (
            df.groupby(['state', 'crop_type'], as_index=False)
            .agg({'production': 'sum'})
        )

        df_heatmap['log_production'] = np.log10(df_heatmap['production'] + 1)

        fig3 = px.density_heatmap(
            df_heatmap,
            x='crop_type',
            y='state',
            z='log_production',
            color_continuous_scale='YlGn',
            hover_data={'production': True},
            title='Crop Production Heatmap by State and Crop Type (Log Scale)'
        )

        fig3.update_layout(
            height=650,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )

        fig3.update_coloraxes(colorbar_title='Log10(Production)')

        st.plotly_chart(fig3, use_container_width=True)
# ==================================================
# PAGE 3 – CLUSTERING
# ==================================================
elif nav == "Cluster Insights":

    st.title("State Clustering by Agricultural Profile")
    set_background("https://64.media.tumblr.com/7e5be0b460f1404bfbf24807efa95f04/5bdfeadfc689526d-6d/s400x600/a87a377cee60d959ae9560c588ec691a2da470db.gif")

    st.markdown("""
    <style>

    /* White glass container */
    [class*="st-key-eda_"], 
    [class*="st-key-cluster_"] {

        background: rgba(255,255,255,0.40);
        backdrop-filter: blur(8px);

        padding: 25px;
        border-radius: 16px;

        border: 1px solid rgba(255,255,255,0.6);

        box-shadow: 
            0px 6px 25px rgba(0,0,0,0.35);

        transition: all 0.25s ease;
      }

        /* Hover animation */
    [class*="st-key-eda_"]:hover,
    [class*="st-key-cluster_"]:hover {

        transform: translateY(-4px);
        box-shadow: 
            0px 10px 30px rgba(0,0,0,0.45);

    }

    </style>
    """, unsafe_allow_html=True)

    with st.container(key="cluster_scatter"):

        fig = px.scatter(
            cluster_df,
            x="avg_production",
            y="avg_growth_rate",
            color="cluster_label",
            size="variability",
            hover_name="state",
            title="Clustered States by Production & Growth"
        )

        fig.update_xaxes(type="log")
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(fig, use_container_width=True)

    with st.container(key="cluster_info"):

        st.markdown("""
        ### Cluster Label Explanation

        **Cluster 0 — Flower Specialized**  
        States focusing mainly on flower production.

        **Cluster 1 — Mixed Crop**  
        States producing a variety of crops.

        **Cluster 2 — Flower + Rice Focused**  
        States with strong production in both flowers and rice.
        """)

# ==================================================
# PAGE 4 – ML PREDICTION
# ==================================================
)
elif nav == "Prediction":

    st.markdown("""
    <style>

    /* MAIN DASHBOARD CONTAINER */
    .st-key-main_container{
        padding:30px;
        border-radius:15px;
        backdrop-filter: blur(6px);
    }

    /* SIDEBAR PANEL */
    .st-key-sidebar{
        background-color:#062906;
        padding:25px;
        border-radius:12px;
        min-height:450px;
        color:white;
    }

    /* MAIN CONTENT PANEL */
    .st-key-mainpanel{
        background-color:rgba(0,0,0,0);
        padding:25px;
        border-radius:12px;
        min-height:450px;
        color:white;
    }

    /* subtle card for results */
    .result-card{
        background:rgba(255,255,255,0.08);
        padding:20px;
        border-radius:12px;
        border:1px solid rgba(255,255,255,0.15);
        font-size:22px;
        text-align:center;
    }

    </style>
    """, unsafe_allow_html=True)

    set_background("https://64.media.tumblr.com/7e5be0b460f1404bfbf24807efa95f04/5bdfeadfc689526d-6d/s400x600/a87a377cee60d959ae9560c588ec691a2da470db.gif")

    st.title("🌾 Crop Production Prediction")

    with st.container(key="main_container"):

        sidebar, main_page = st.columns([1,2])

        # =========================
        # SIDEBAR
        # =========================
        with sidebar:
            with st.container(key="sidebar"):

                st.subheader("Fill details below")

                crop = st.selectbox(
                    "Crop Type",
                    sorted(df["crop_type"].unique())
                )

                state = st.selectbox(
                    "State",
                    sorted(df["state"].unique())
                )

                planted_area_input = st.text_input(
                    "Planted Area (hectares)",
                    "1000"
                )

                precipitation_input = st.text_input(
                    "Precipitation (mm)",
                    "200"
                )

                year = st.slider(
                    "Year",
                    int(df["year"].min()),
                    int(df["year"].max()+3),
                    2023
                )

                predict_button = st.button("Predict Production")

        # =========================
        # MAIN PAGE
        # =========================
        with main_page:
            with st.container(key="mainpanel"):

                st.subheader("Prediction Model Info")

                st.write("Algorithm:", config["model"]["algorithm_pred"])
                st.write("R² Score:", config["model"]["r_square_score"])
                st.write("RMSE:", config["model"]["rmse"])
                st.write("MAE:", config["model"]["MAE"])

                result_placeholder = st.empty()

                if predict_button:

                    try:

                        planted_area = float(planted_area_input)
                        precipitation = float(precipitation_input)

                        input_df = pd.DataFrame({
                            "state":[state],
                            "crop_type":[crop],
                            "planted_area":[planted_area],
                            "precipitation":[precipitation],
                            "year":[year]
                        })

                        log_prediction = prediction_model.predict(input_df)

                        prediction = np.expm1(log_prediction[0])

                        result_placeholder.markdown(
                            f"""
                            <div class="result-card">
                            🌾 Predicted Production<br><br>
                            <b>{prediction:,.2f}</b>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                    except ValueError:
                        result_placeholder.error("Invalid numeric input.")
# =================================================
# ML FORECASTING
# =================================================
elif nav == "Forecasting":

    st.markdown("""
    <style>

    /* MAIN DASHBOARD CONTAINER */
    .st-key-main_container{
        padding:30px;
        border-radius:15px;
        backdrop-filter: blur(6px);
    }

    /* SIDEBAR PANEL */
    .st-key-sidebar{
        background-color:#062906;
        padding:25px;
        border-radius:12px;
        min-height:450px;
        color:white;
    }

    /* MAIN CONTENT PANEL */
    .st-key-mainpanel{
        background-color:rgba(0,0,0,0);
        padding:25px;
        border-radius:12px;
        min-height:450px;
        color:white;
    }

    /* subtle card for results */
    .result-card{
        background:rgba(255,255,255,0.08);
        padding:20px;
        border-radius:12px;
        border:1px solid rgba(255,255,255,0.15);
        font-size:22px;
        text-align:center;
    }

    </style>
    """, unsafe_allow_html=True)

    set_background("https://64.media.tumblr.com/7e5be0b460f1404bfbf24807efa95f04/5bdfeadfc689526d-6d/s400x600/a87a377cee60d959ae9560c588ec691a2da470db.gif")

    st.title("📈 Crop Production Forecasting")

    with st.container(key="main_container"):

        sidebar, main_page = st.columns([1,2])

        # =========================
        # SIDEBAR
        # =========================
        with sidebar:
            with st.container(key="sidebar"):

                st.subheader("Fill details below")

                crop_list = list(prophet_models.keys())

                selected_crop = st.selectbox(
                    "Select Crop",
                    crop_list
                )

                forecast_years = st.slider(
                    "Forecast Years",
                    1,
                    10,
                    3
                )

                forecast_button = st.button("Generate Forecast")

        # =========================
        # MAIN PANEL
        # =========================
        with main_page:
            with st.container(key="mainpanel"):

                st.subheader("Forecasting Model Info")

                st.write("Algorithm:",config["model"]["algorithm_forecast"])
                st.write("RMSE:",config["model"]["rmse_forecast"])
                st.write("MAPE:",config["model"]["MAPE"])

                if forecast_button:

                    model = prophet_models[selected_crop]

                    periods = forecast_years * 12

                    future = model.make_future_dataframe(
                        periods=periods,
                        freq="M"
                    )

                    forecast = model.predict(future)

                    st.subheader("Forecast Result")

                    fig = go.Figure()

                    fig.add_trace(go.Scatter(
                        x=forecast["ds"],
                        y=forecast["yhat"],
                        name="Forecast"
                    ))

                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )
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
