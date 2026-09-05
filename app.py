
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
        .main-title {
            font-size: 42px;
            font-weight: 700;
            text-align: center;
            margin-bottom: 5px;
        }

        .subtitle {
            text-align: center;
            font-size: 18px;
            color: #666;
            margin-bottom: 30px;
        }

        .prediction-box {
            padding: 25px;
            border-radius: 15px;
            text-align: center;
            margin-top: 20px;
            border: 1px solid #ddd;
        }

        .prediction-price {
            font-size: 38px;
            font-weight: bold;
        }

        .info-box {
            padding: 15px;
            border-radius: 10px;
            background-color: #f5f5f5;
            margin-top: 20px;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent



MODEL_PATH = BASE_DIR / "models" / "car_price_model.pkl"
COLUMNS_PATH = BASE_DIR / "models" / "columns.pkl"
DATA_PATH = BASE_DIR / "data" / "car.csv"


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


# ============================================================
# LOAD TRAINING COLUMNS
# ============================================================

@st.cache_resource
def load_columns():
    return joblib.load(COLUMNS_PATH)


# ============================================================
# LOAD CAR DATA
# ============================================================

@st.cache_data
def load_car_data():
    return pd.read_csv(DATA_PATH)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🚗 Car Price Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Predict the estimated price of a used car using Machine Learning'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# ERROR HANDLING - FILES
# ============================================================

try:

    if not MODEL_PATH.exists():
        st.error(
            f"❌ Model file not found:\n\n`{MODEL_PATH}`"
        )
        st.stop()

    if not COLUMNS_PATH.exists():
        st.error(
            f"❌ Columns file not found:\n\n`{COLUMNS_PATH}`"
        )
        st.stop()

    if not DATA_PATH.exists():
        st.error(
            f"❌ Dataset not found:\n\n`{DATA_PATH}`"
        )
        st.stop()

    model = load_model()
    expected_columns = load_columns()
    df = load_car_data()

except Exception as e:

    st.error("❌ Error while loading project files.")
    st.exception(e)
    st.stop()


# ============================================================
# VALIDATE DATASET
# ============================================================

required_columns = [
    "brand",
    "model",
    "year",
    "transmission",
    "mileage",
    "fuelType",
    "tax",
    "mpg",
    "engineSize"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:

    st.error(
        "❌ The following required columns are missing "
        "from `car.csv`:"
    )

    st.write(missing_columns)

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("🚗 Car Details")

    st.write(
        "Enter the details of the car to estimate "
        "its selling price."
    )

    st.divider()

    st.info(
        "The model uses the same one-hot encoding "
        "structure used during training."
    )


# ============================================================
# INPUT SECTION
# ============================================================

st.subheader("Enter Car Information")

col1, col2 = st.columns(2)


# ------------------------------------------------------------
# BRAND
# ------------------------------------------------------------

with col1:

    brands = sorted(
        df["brand"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    brand = st.selectbox(
        "🏷️ Brand",
        brands
    )


# ------------------------------------------------------------
# MODEL
# ------------------------------------------------------------

with col2:

    # Show models belonging to selected brand
    brand_models = (
        df[df["brand"].astype(str) == brand]["model"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    brand_models = sorted(brand_models)

    if len(brand_models) == 0:

        models = sorted(
            df["model"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

    else:
        models = brand_models

    model_name = st.selectbox(
        "🚘 Model",
        models
    )


# ============================================================
# YEAR AND TRANSMISSION
# ============================================================

col3, col4 = st.columns(2)


with col3:

    min_year = int(df["year"].min())
    max_year = int(df["year"].max())

    year = st.number_input(
        "📅 Year",
        min_value=min_year,
        max_value=max_year,
        value=max_year,
        step=1
    )


with col4:

    transmissions = sorted(
        df["transmission"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    transmission = st.selectbox(
        "⚙️ Transmission",
        transmissions
    )


# ============================================================
# MILEAGE AND FUEL TYPE
# ============================================================

col5, col6 = st.columns(2)


with col5:

    mileage_min = float(df["mileage"].min())
    mileage_max = float(df["mileage"].max())

    mileage = st.number_input(
        "🛣️ Mileage",
        min_value=0.0,
        max_value=max(mileage_max, 1.0),
        value=float(
            min(
                max(
                    df["mileage"].median(),
                    0
                ),
                max(mileage_max, 1.0)
            )
        ),
        step=100.0
    )


with col6:

    fuel_types = sorted(
        df["fuelType"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    fuel_type = st.selectbox(
        "⛽ Fuel Type",
        fuel_types
    )


# ============================================================
# TAX AND MPG
# ============================================================

col7, col8 = st.columns(2)


with col7:

    tax_max = float(df["tax"].max())

    tax = st.number_input(
        "💰 Tax",
        min_value=0.0,
        max_value=max(tax_max, 1.0),
        value=float(
            max(
                df["tax"].median(),
                0
            )
        ),
        step=1.0
    )


with col8:

    mpg_max = float(df["mpg"].max())

    mpg = st.number_input(
        "⛽ MPG",
        min_value=0.0,
        max_value=max(mpg_max, 1.0),
        value=float(
            max(
                df["mpg"].median(),
                0
            )
        ),
        step=0.1
    )


# ============================================================
# ENGINE SIZE
# ============================================================

engine_max = float(df["engineSize"].max())

engine_size = st.number_input(
    "🔧 Engine Size",
    min_value=0.0,
    max_value=max(engine_max, 1.0),
    value=float(
        max(
            df["engineSize"].median(),
            0
        )
    ),
    step=0.1
)


# ============================================================
# PREDICTION BUTTON
# ============================================================

st.divider()

predict_button = st.button(
    "🔮 Predict Car Price",
    type="primary",
    use_container_width=True
)


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    try:

        # ----------------------------------------------------
        # CREATE INPUT DATAFRAME
        # ----------------------------------------------------

        input_data = pd.DataFrame({
            "brand": [brand],
            "model": [model_name],
            "year": [year],
            "transmission": [transmission],
            "mileage": [mileage],
            "fuelType": [fuel_type],
            "tax": [tax],
            "mpg": [mpg],
            "engineSize": [engine_size]
        })


        # ----------------------------------------------------
        # RECREATE ONE-HOT ENCODING
        # ----------------------------------------------------

        input_encoded = pd.get_dummies(
            input_data,
            drop_first=True
        )


        # ----------------------------------------------------
        # ALIGN COLUMNS WITH TRAINING DATA
        # ----------------------------------------------------

        input_encoded = input_encoded.reindex(
            columns=expected_columns,
            fill_value=0
        )


        # ----------------------------------------------------
        # ENSURE NUMERIC DATA
        # ----------------------------------------------------

        input_encoded = input_encoded.apply(
            pd.to_numeric,
            errors="coerce"
        ).fillna(0)


        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        prediction = model.predict(input_encoded)

        predicted_price = float(prediction[0])


        # ----------------------------------------------------
        # DISPLAY RESULT
        # ----------------------------------------------------

        st.success("✅ Prediction completed successfully!")

        # Convert price into lakh
        price_lakh = predicted_price / 100000

        st.markdown(
            f"""
            <div class="prediction-box">

                <div style="font-size:20px;">
                    Estimated Car Price
                </div>

                <div class="prediction-price">
                    ₹{price_lakh:.2f} Lakh
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )
        


        # ----------------------------------------------------
        # DISPLAY INPUT SUMMARY
        # ----------------------------------------------------

        st.subheader("📋 Car Details Used for Prediction")

        result_col1, result_col2 = st.columns(2)

        with result_col1:

            st.write(f"**Brand:** {brand}")
            st.write(f"**Model:** {model_name}")
            st.write(f"**Year:** {year}")
            st.write(f"**Transmission:** {transmission}")
            st.write(f"**Mileage:** {mileage:,.0f}")

        with result_col2:

            st.write(f"**Fuel Type:** {fuel_type}")
            st.write(f"**Tax:** {tax:,.0f}")
            st.write(f"**MPG:** {mpg:.2f}")
            st.write(f"**Engine Size:** {engine_size:.2f}")


    except Exception as e:

        st.error(
            "❌ An error occurred while making the prediction."
        )

        st.exception(e)


# # ============================================================
# # MODEL INFORMATION
# # ============================================================

# st.divider()

# st.subheader("ℹ️ About This Model")

# st.markdown(
#     """
#     This application uses the saved **Linear Regression** model
#     `car_price_model.pkl`.

#     ### Preprocessing

#     The input is converted using:

#     - `pd.get_dummies(drop_first=True)`
#     - Columns are aligned with `columns.pkl`
#     - Missing training columns are filled with `0`

#     ### Important

#     **`scaler.pkl` is NOT used in this application.**

#     The selected Linear Regression model was trained on the
#     **unscaled one-hot encoded X**, so applying a scaler during
#     prediction would make the input inconsistent with the training
#     data.

#     Therefore, the prediction pipeline is:

#     **User Input → One-Hot Encoding → Column Alignment → Linear Regression → Price**
#     """
# )

# ============================================================
# FOOTER
# ============================================================

st.caption(
    "🚗 Car Price Prediction | Machine Learning Project"
)