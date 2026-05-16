from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


ROOT_DIR = Path(__file__).resolve().parent
MODEL_PATH = ROOT_DIR / "models" / "weather_xgboost.joblib"


st.set_page_config(
    page_title="Prediksi Cuaca XGBoost",
    page_icon=None,
    layout="wide",
)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def numeric_input(column: str, metadata: dict) -> float:
    values = metadata["numeric"][column]
    minimum = float(values["min"])
    maximum = float(values["max"])
    default = float(values["mean"])
    span = maximum - minimum
    step = 1.0 if span > 20 else 0.1
    return st.number_input(
        column,
        min_value=minimum,
        max_value=maximum,
        value=round(default, 2),
        step=step,
    )


def main() -> None:
    st.title("Prediksi Tipe Cuaca")

    if not MODEL_PATH.exists():
        st.error("Model belum tersedia. Jalankan `python src/train_model.py` terlebih dahulu.")
        st.stop()

    artifact = load_model()
    pipeline = artifact["pipeline"]
    label_encoder = artifact["label_encoder"]
    metadata = artifact["feature_metadata"]
    metrics = artifact.get("metrics", {})

    with st.sidebar:
        st.header("Input Kondisi")
        user_input = {}
        for column in metadata["numeric"]:
            user_input[column] = numeric_input(column, metadata)

        for column, options in metadata["categorical"].items():
            user_input[column] = st.selectbox(column, options=options)

    input_df = pd.DataFrame([user_input], columns=metadata["feature_columns"])
    encoded_prediction = int(pipeline.predict(input_df)[0])
    prediction = label_encoder.inverse_transform([encoded_prediction])[0]
    probabilities = pipeline.predict_proba(input_df)[0]
    probability_df = (
        pd.DataFrame(
            {
                "Weather Type": label_encoder.classes_,
                "Probability": probabilities,
            }
        )
        .sort_values("Probability", ascending=False)
        .reset_index(drop=True)
    )

    top_probability = float(probability_df.iloc[0]["Probability"])

    col_prediction, col_metric = st.columns([2, 1])
    with col_prediction:
        st.subheader("Hasil Prediksi")
        st.metric("Weather Type", prediction, f"{top_probability:.1%} confidence")

    with col_metric:
        st.subheader("Evaluasi Model")
        st.metric("F1-macro", f"{metrics.get('f1_macro', 0):.3f}")
        st.metric("Accuracy", f"{metrics.get('accuracy', 0):.3f}")

    st.subheader("Probabilitas per Kelas")
    st.bar_chart(probability_df.set_index("Weather Type"))

    st.subheader("Data Input")
    st.dataframe(input_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
