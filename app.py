import pickle
import numpy as np
import streamlit as st
import tensorflow as tf

st.set_page_config(
    page_title="Breast Cancer Diagnostic System", page_icon="🔬", layout="wide"
)

@st.cache_resource
def load_artifacts():
  model = tf.keras.models.load_model("breast_cancer_model.keras")
  with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
  return model, scaler


model, scaler = load_artifacts()

st.title("🔬 Breast Cancer Diagnostic Predictor")
st.markdown(
    "Enter the cytological numerical measurements below or use the sidebar presets to test the system."
)

st.sidebar.header("🧪 Test Presets")
st.sidebar.markdown("Load test measurements to test predictions quickly:")

malignant_example = (
    "17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419,"
    " 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373,"
    " 0.01587, 0.03003, 0.006193, 25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656,"
    " 0.7119, 0.2654, 0.4601, 0.1189"
)

benign_example = (
    "13.54, 14.36, 87.46, 566.3, 0.09779, 0.08129, 0.06664, 0.04781, 0.1885,"
    " 0.05766, 0.2699, 0.7886, 2.058, 23.56, 0.008462, 0.0146, 0.02387,"
    " 0.01315, 0.0198, 0.0023, 15.11, 19.26, 99.7, 711.2, 0.144, 0.1773,"
    " 0.239, 0.1288, 0.2977, 0.07259"
)

if "input_text" not in st.session_state:
  st.session_state.input_text = ""

col_s1, col_s2 = st.sidebar.columns(2)
if col_s1.button("Load Malignant Sample"):
  st.session_state.input_text = malignant_example
if col_s2.button("Load Benign Sample"):
  st.session_state.input_text = benign_example

if st.sidebar.button("Clear Input"):
  st.session_state.input_text = ""

user_input = st.text_area(
    label="Enter 30 comma-separated feature values:",
    value=st.session_state.input_text,
    placeholder="e.g. 17.99, 10.38, 122.8, 1001.0, 0.1184, ...",
    height=140,
)

predict_btn = st.button("Run Prediction", type="primary")

if predict_btn:
  cleaned_text = user_input.strip()

  if not cleaned_text:
    st.warning("Please enter feature values before predicting.")
  else:
    try:
      raw_items = [x.strip() for x in cleaned_text.split(",") if x.strip()]
      feature_values = [float(x) for x in raw_items]

      if len(feature_values) != 30:
        st.error(
            f"Expected exactly **30** values, but detected **{len(feature_values)}**. Please check your input."
        )
      else:
        input_array = np.asarray(feature_values, dtype=np.float32).reshape(
            1, -1
        )
        scaled_input = scaler.transform(input_array)

        prediction_probs = model.predict(scaled_input)
        predicted_class = int(np.argmax(prediction_probs[0]))
        confidence = float(np.max(prediction_probs[0])) * 100

        st.divider()
        st.subheader("Diagnostic Result")

        if predicted_class == 0:
          st.error(
              f"🚨 **Diagnosis: Malignant (Cancerous)**\n\n"
              f"**Model Confidence:** `{confidence:.2f}%`"
          )
        else:
          st.success(
              f"✅ **Diagnosis: Benign (Non-Cancerous)**\n\n"
              f"**Model Confidence:** `{confidence:.2f}%`"
          )

    except ValueError:
      st.error(
          "Invalid input detected. Please ensure all 30 values are valid numbers separated by commas."
      )