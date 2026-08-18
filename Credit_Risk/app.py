import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

from pathlib import Path


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"


# ============================================================
# 2. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Credit Risk Predictor",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 3. BRAND STYLING
# ============================================================
# Same navy / gold palette used across the InsightByFrank guides,
# so the app and the written material feel like one product.

st.markdown(
    """
    <style>
    :root{
        --navy:#0b1f3a;
        --blue:#1e5fbf;
        --blue-light:#eaf1fc;
        --gold:#c9973f;
        --gray-light:#f4f6f9;
    }

    .stApp{
        background-color:#ffffff;
        color:#1c2531;
    }

    /* Force dark, readable text in the main content area regardless of the
       viewer's light/dark system theme -- the background above is always
       white, so text must always be dark to stay legible on top of it. */
    .main, .main p, .main li, .main span, .main label,
    .main h1, .main h2, .main h3, .main h4, .main h5, .main h6,
    .main .stMarkdown, div[data-testid="stMetricLabel"],
    div[data-testid="stMetricValue"]{
        color:#1c2531 !important;
    }

    /* The hero banner and colored risk cards keep their own white text --
       they sit on dark/colored backgrounds, so this override restores that. */
    .hero-banner, .hero-banner *,
    .risk-card, .risk-card *{
        color:#ffffff !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"]{
        background-color:var(--navy);
    }
    section[data-testid="stSidebar"] *{
        color:#eaf1fc !important;
    }
    section[data-testid="stSidebar"] .stSlider label,
    section[data-testid="stSidebar"] .stNumberInput label{
        font-weight:600;
        color:#ffffff !important;
    }

    /* Hero banner */
    .hero-banner{
        background:linear-gradient(120deg, var(--navy) 0%, var(--blue) 100%);
        padding:26px 32px;
        border-radius:14px;
        color:#ffffff;
        margin-bottom:24px;
    }
    .hero-banner h1{
        margin:0 0 4px 0;
        font-size:30px;
        color:#ffffff;
    }
    .hero-banner p{
        margin:0;
        color:#cfe0f7;
        font-size:15px;
    }
    .hero-tag{
        display:inline-block;
        background:var(--gold);
        color:#3a2900;
        font-size:11.5px;
        font-weight:700;
        letter-spacing:1px;
        text-transform:uppercase;
        padding:4px 12px;
        border-radius:20px;
        margin-bottom:10px;
    }

    /* Result cards */
    .risk-card{
        border-radius:12px;
        padding:20px 22px;
        text-align:center;
        color:#ffffff;
        font-weight:700;
    }
    .risk-high{ background:linear-gradient(135deg,#8a1f1f,#c0392b); }
    .risk-moderate{ background:linear-gradient(135deg,#8a5c00,#c9973f); }
    .risk-low{ background:linear-gradient(135deg,#155724,#2e8b57); }

    .example-caption{
        font-size:12.5px;
        color:#b9cdec;
        margin-top:-6px;
        margin-bottom:14px;
    }

    /* Section divider labels */
    .section-label{
        font-size:12px;
        letter-spacing:1.5px;
        text-transform:uppercase;
        color:var(--gold);
        font-weight:700;
        margin:18px 0 4px 0;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 4. LOAD MODEL ARTIFACTS
# ============================================================

@st.cache_resource
def load_artifacts():

    model = joblib.load(
        MODEL_DIR / "xgb_credit_model.pkl"
    )

    imputer = joblib.load(
        MODEL_DIR / "final_imputer.pkl"
    )

    feature_columns = joblib.load(
        MODEL_DIR / "feature_columns.pkl"
    )

    return model, imputer, feature_columns


model, imputer, feature_columns = load_artifacts()


# ============================================================
# 5. CREATE SHAP EXPLAINER
# ============================================================

explainer = shap.TreeExplainer(
    model,
    feature_perturbation="tree_path_dependent"
)


# ============================================================
# 6. EXAMPLE BORROWER PROFILES
# ============================================================
# Lets a visitor explore the app in one click instead of typing
# ten values before seeing anything happen.

EXAMPLE_PROFILES = {
    "🟢 Low-risk example": {
        "age": 45, "monthly_income": 8000.0, "dependents": 1,
        "revolving_util": 0.15, "debt_ratio": 0.20,
        "open_credit_lines": 8, "real_estate_loans": 2,
        "late_30_59": 0, "late_60_89": 0, "late_90": 0,
    },
    "🟡 Moderate-risk example": {
        "age": 35, "monthly_income": 4000.0, "dependents": 2,
        "revolving_util": 0.55, "debt_ratio": 0.45,
        "open_credit_lines": 5, "real_estate_loans": 1,
        "late_30_59": 1, "late_60_89": 0, "late_90": 0,
    },
    "🔴 High-risk example": {
        "age": 28, "monthly_income": 2200.0, "dependents": 3,
        "revolving_util": 0.95, "debt_ratio": 0.75,
        "open_credit_lines": 3, "real_estate_loans": 0,
        "late_30_59": 2, "late_60_89": 1, "late_90": 2,
    },
}


def load_profile(name):
    profile = EXAMPLE_PROFILES[name]
    for key, value in profile.items():
        st.session_state[key] = value


# ============================================================
# 7. HERO HEADER
# ============================================================

st.markdown(
    """
    <div class="hero-banner">
        <div class="hero-tag">Learn Data Analytics with InsightByFrank</div>
        <h1>💳 Credit Risk Predictor</h1>
        <p>Estimate the probability that a borrower falls seriously behind on payments
        within two years — and see exactly which factors drove that prediction.</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.info(
    "This application is for educational purposes only. "
    "It must not be used as a real credit underwriting "
    "or lending decision system."
)


# ============================================================
# 8. SIDEBAR — ORGANIZED, WITH QUICK EXAMPLES
# ============================================================

st.sidebar.markdown("## 👤 Borrower Profile")
st.sidebar.markdown(
    "<p class='example-caption'>New here? Load an example to see the app in action.</p>",
    unsafe_allow_html=True
)

example_cols = st.sidebar.columns(3)
for col, name in zip(example_cols, EXAMPLE_PROFILES.keys()):
    col.button(
        name.split(" ")[0],
        help=name,
        on_click=load_profile,
        args=(name,),
        width="stretch"
    )

st.sidebar.divider()

# Set each widget's starting value exactly once, the first time the app
# runs. After that, Streamlit's session state (and the example-profile
# buttons above) fully own these values — a widget must never be given
# both a `value=` and a `key=` that's already in session state, or the
# two sources of truth conflict.
_DEFAULTS = {
    "age": 40, "dependents": 0, "monthly_income": 5000.0,
    "debt_ratio": 0.30, "revolving_util": 0.30,
    "open_credit_lines": 5, "real_estate_loans": 1,
    "late_30_59": 0, "late_60_89": 0, "late_90": 0,
}
for _key, _default in _DEFAULTS.items():
    st.session_state.setdefault(_key, _default)

st.sidebar.markdown("<div class='section-label'>Personal</div>", unsafe_allow_html=True)

age = st.sidebar.slider(
    "Age",
    min_value=18,
    max_value=100,
    key="age"
)

dependents = st.sidebar.number_input(
    "Number of Dependents",
    min_value=0,
    step=1,
    key="dependents"
)

st.sidebar.markdown("<div class='section-label'>Income &amp; Debt</div>", unsafe_allow_html=True)

monthly_income = st.sidebar.number_input(
    "Monthly Income ($)",
    min_value=0.0,
    step=100.0,
    key="monthly_income"
)

debt_ratio = st.sidebar.slider(
    "Debt Ratio",
    min_value=0.0,
    max_value=2.0,
    step=0.01,
    key="debt_ratio"
)

revolving_util = st.sidebar.slider(
    "Revolving Utilization",
    min_value=0.0,
    max_value=2.0,
    step=0.01,
    key="revolving_util",
    help="Share of available revolving credit currently in use. Above 1.0 means over the limit."
)

st.sidebar.markdown("<div class='section-label'>Credit Accounts</div>", unsafe_allow_html=True)

open_credit_lines = st.sidebar.number_input(
    "Open Credit Lines / Loans",
    min_value=0,
    step=1,
    key="open_credit_lines"
)

real_estate_loans = st.sidebar.number_input(
    "Real Estate Loans / Lines",
    min_value=0,
    step=1,
    key="real_estate_loans"
)

st.sidebar.markdown("<div class='section-label'>Payment History</div>", unsafe_allow_html=True)

late_30_59 = st.sidebar.number_input(
    "30–59 Days Late",
    min_value=0,
    step=1,
    key="late_30_59"
)

late_60_89 = st.sidebar.number_input(
    "60–89 Days Late",
    min_value=0,
    step=1,
    key="late_60_89"
)

late_90 = st.sidebar.number_input(
    "90+ Days Late",
    min_value=0,
    step=1,
    key="late_90"
)

st.sidebar.divider()

predict_button = st.sidebar.button(
    "🔮 Predict Risk",
    type="primary",
    width="stretch"
)


# ============================================================
# 9. RISK GAUGE HELPER
# ============================================================

def render_risk_gauge(probability, moderate_cutoff, high_cutoff):

    # A half-donut "speedometer" built with matplotlib only -- no extra
    # dependency beyond what the app already imports.
    fig, ax = plt.subplots(figsize=(5, 3), subplot_kw={"aspect": "equal"})

    zones = [
        (0, moderate_cutoff, "#2e8b57"),
        (moderate_cutoff, high_cutoff, "#c9973f"),
        (high_cutoff, 1.0, "#c0392b"),
    ]

    for start, end, color in zones:
        ax.pie(
            [end - start],
            radius=1,
            startangle=180 - (start * 180),
            counterclock=False,
            colors=[color],
            wedgeprops={"width": 0.35, "edgecolor": "white"},
        )

    # Needle showing the actual probability
    needle_angle = np.radians(180 - probability * 180)
    needle_len = 0.85
    ax.plot(
        [0, needle_len * np.cos(needle_angle)],
        [0, needle_len * np.sin(needle_angle)],
        color="#0b1f3a", linewidth=3, solid_capstyle="round", zorder=5,
    )
    ax.scatter([0], [0], color="#0b1f3a", s=110, zorder=6)

    ax.text(
        0, -0.25, f"{probability:.1%}",
        ha="center", va="center", fontsize=26, fontweight="bold", color="#0b1f3a",
    )
    ax.text(
        0, -0.45, "Default probability",
        ha="center", va="center", fontsize=10, color="#5b6675",
    )

    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-0.55, 1.15)
    ax.axis("off")
    fig.patch.set_alpha(0)

    return fig


# ============================================================
# 10. PREDICTION
# ============================================================

if predict_button:

    # --------------------------------------------------------
    # Create borrower dataframe
    # --------------------------------------------------------

    row = pd.DataFrame([{

        "RevolvingUtilizationOfUnsecuredLines":
            revolving_util,

        "age":
            age,

        "NumberOfTime30-59DaysPastDueNotWorse":
            late_30_59,

        "DebtRatio":
            debt_ratio,

        "MonthlyIncome":
            monthly_income,

        "NumberOfOpenCreditLinesAndLoans":
            open_credit_lines,

        "NumberOfTimes90DaysLate":
            late_90,

        "NumberRealEstateLoansOrLines":
            real_estate_loans,

        "NumberOfTime60-89DaysPastDueNotWorse":
            late_60_89,

        "NumberOfDependents":
            dependents,

        "MonthlyIncome_missing":
            0,

        "NumberOfDependents_missing":
            0
    }])


    # ========================================================
    # 11. FEATURE ENGINEERING
    # ========================================================

    row["TotalTimesPastDue"] = (
        row[
            "NumberOfTime30-59DaysPastDueNotWorse"
        ]
        +
        row[
            "NumberOfTime60-89DaysPastDueNotWorse"
        ]
        +
        row[
            "NumberOfTimes90DaysLate"
        ]
    )


    row["EstimatedMonthlyDebtPayment"] = (
        row["DebtRatio"]
        *
        row["MonthlyIncome"]
    )


    row["IncomePerDependent"] = (
        row["MonthlyIncome"]
        /
        (row["NumberOfDependents"] + 1)
    )


    row["CreditLinesPerAge"] = (
        row["NumberOfOpenCreditLinesAndLoans"]
        /
        row["age"].replace(0, np.nan)
    )


    row["IsOverutilized"] = (
        row[
            "RevolvingUtilizationOfUnsecuredLines"
        ] > 1
    ).astype(int)


    # ========================================================
    # 12. ALIGN FEATURES
    # ========================================================

    try:

        row = row[feature_columns]

    except KeyError as error:

        st.error(
            "The input features do not match the "
            "features used to train the model."
        )

        st.exception(error)

        st.stop()


    # ========================================================
    # 13. IMPUTE MISSING VALUES
    # ========================================================

    try:

        row_imputed = pd.DataFrame(
            imputer.transform(row),
            columns=feature_columns,
            index=row.index
        )

    except Exception as error:

        st.error(
            "The preprocessing step failed."
        )

        st.exception(error)

        st.stop()


    # ========================================================
    # 14. SAFETY CHECK
    # ========================================================

    if row_imputed.isnull().sum().sum() > 0:

        st.error(
            "Missing values remain after imputation."
        )

        st.stop()


    # ========================================================
    # 15. MODEL PREDICTION
    # ========================================================

    with st.spinner("Scoring borrower..."):

        try:

            probability = float(
                model.predict_proba(
                    row_imputed
                )[0, 1]
            )

            predicted_class = int(
                model.predict(
                    row_imputed
                )[0]
            )

        except Exception as error:

            st.error(
                "The model prediction failed."
            )

            st.exception(error)

            st.stop()


    # ========================================================
    # 16. ADJUSTABLE RISK THRESHOLDS
    # ========================================================

    st.divider()

    with st.expander("⚙️ Adjust risk thresholds (advanced)", expanded=False):
        st.caption(
            "Move these to match your own risk appetite — the gauge and "
            "risk label below update instantly."
        )
        threshold_cols = st.columns(2)
        moderate_cutoff = threshold_cols[0].slider(
            "Moderate risk starts at", 0.05, 0.50, 0.20, 0.05
        )
        high_cutoff = threshold_cols[1].slider(
            "High risk starts at", 0.30, 0.90, 0.50, 0.05
        )
        if moderate_cutoff >= high_cutoff:
            st.warning("Moderate cutoff should be lower than high cutoff — using defaults instead.")
            moderate_cutoff, high_cutoff = 0.20, 0.50


    # ========================================================
    # 17. DISPLAY RESULT
    # ========================================================

    st.subheader("Credit Risk Assessment")

    result_col1, result_col2 = st.columns([1, 1.3])

    with result_col1:
        gauge_fig = render_risk_gauge(probability, moderate_cutoff, high_cutoff)
        st.pyplot(gauge_fig, clear_figure=True)
        plt.close(gauge_fig)

    with result_col2:

        if probability >= high_cutoff:
            card_class, label, icon = "risk-high", "HIGH RISK", "🔴"
        elif probability >= moderate_cutoff:
            card_class, label, icon = "risk-moderate", "MODERATE RISK", "🟡"
        else:
            card_class, label, icon = "risk-low", "LOWER RISK", "🟢"

        st.markdown(
            f"""
            <div class="risk-card {card_class}">
                <div style="font-size:15px; opacity:0.85; letter-spacing:1px;">
                    {icon} MODEL PREDICTION
                </div>
                <div style="font-size:34px; margin:8px 0;">{label}</div>
                <div style="font-size:15px; font-weight:400;">
                    Default probability: <strong>{probability:.1%}</strong><br>
                    Model classification:
                    <strong>{"Potential Default" if predicted_class == 1 else "No Default"}</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("")
        metric_cols = st.columns(3)
        metric_cols[0].metric("Age", f"{age}")
        metric_cols[1].metric("Monthly Income", f"${monthly_income:,.0f}")
        metric_cols[2].metric("Times Past Due", f"{late_30_59 + late_60_89 + late_90}")


    # ========================================================
    # 18. SHAP EXPLANATION
    # ========================================================

    st.divider()

    st.subheader("🔎 Why did the model make this prediction?")

    st.caption(
        "SHAP shows how individual features contributed "
        "to this prediction — bars pushing right increase risk, "
        "bars pushing left decrease it."
    )

    try:

        shap_result = explainer(
            row_imputed
        )

        fig = plt.figure(
            figsize=(10, 6)
        )

        shap.waterfall_plot(
            shap_result[0],
            max_display=10,
            show=False
        )

        st.pyplot(
            fig,
            clear_figure=True
        )

        plt.close(fig)

    except Exception as error:

        st.warning(
            "The prediction was successful, but the "
            "SHAP explanation could not be generated."
        )

        st.exception(error)


    # ========================================================
    # 19. PROCESSED FEATURES
    # ========================================================

    st.divider()

    with st.expander("📋 View processed features"):

        processed_features = (
            row_imputed
            .T
            .rename(
                columns={0: "Value"}
            )
        )

        st.dataframe(
            processed_features,
            width="stretch"
        )


# ============================================================
# 20. INITIAL SCREEN
# ============================================================

else:

    welcome_cols = st.columns(3)

    with welcome_cols[0]:
        st.markdown("### 1️⃣ Set the profile")
        st.write(
            "Use the sidebar to describe a borrower, or click one of the "
            "three example buttons at the top of the sidebar to load a "
            "ready-made profile instantly."
        )

    with welcome_cols[1]:
        st.markdown("### 2️⃣ Predict")
        st.write(
            "Click **🔮 Predict Risk**. The model scores the borrower and "
            "shows a live risk gauge, a color-coded risk card, and the "
            "exact probability of default."
        )

    with welcome_cols[2]:
        st.markdown("### 3️⃣ Understand why")
        st.write(
            "A SHAP waterfall chart breaks the prediction down feature by "
            "feature, so you can see exactly what pushed the score up or down."
        )

    st.divider()
    st.markdown(
        "👈 **Try it now** — click a colored example button at the top of the sidebar, "
        "then hit **Predict Risk**."
    )