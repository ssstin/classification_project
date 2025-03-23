import streamlit as st
import joblib
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Set page title
st.title("PAD vs. PMRI Classification Tool")
st.subheader("Predicting impaired blood microcirculation from scan measurements")

# Add brief instructions
st.write("""
This tool helps classify patients as either healthy or having impaired blood microcirculation (PAD).
Enter the measurement values from both baseline and high-quality scans to get a prediction.
""")

# Create two columns for input
col1, col2 = st.columns(2)

# Baseline measurements (left column)
with col1:
    st.subheader("Baseline Scan Measurements")
    adc_right_baseline = st.number_input("ADC Right Baseline", min_value=0.5, max_value=3.0, value=1.45, step=0.01)
    adc_left_baseline = st.number_input("ADC Left Baseline", min_value=0.5, max_value=3.0, value=1.38, step=0.01)
    pf_right_baseline = st.number_input("Pf Right Baseline", min_value=0.1, max_value=25.0, value=7.2, step=0.1)
    pd_right_baseline = st.number_input("Pd Right Baseline", min_value=0.5, max_value=2.0, value=1.34, step=0.01)

# High-quality measurements (right column)
with col2:
    st.subheader("High-Quality Scan Measurements")
    adc_right_high_quality = st.number_input("ADC Right High Quality", min_value=0.5, max_value=3.0, value=1.52, step=0.01)
    adc_left_high_quality = st.number_input("ADC Left High Quality", min_value=0.5, max_value=3.0, value=1.43, step=0.01)
    pf_right_high_quality = st.number_input("Pf Right High Quality", min_value=0.1, max_value=25.0, value=8.5, step=0.1)
    pd_right_high_quality = st.number_input("Pd Right High Quality", min_value=0.5, max_value=2.0, value=1.36, step=0.01)

# Add a predict button
if st.button("Predict"):
    # This section would normally load your saved model and scaler
    # For demonstration, we'll assume they're loaded here
    try:
        # Load the model and scaler
        model = joblib.load('knn_model.joblib')
        scaler = joblib.load('scaler.joblib')
        top_features = joblib.load('top_features.joblib')
        
        # Calculate derived features
        features = {}
        
        # Averages
        features['adc_right_avg'] = (adc_right_baseline + adc_right_high_quality) / 2
        features['adc_left_avg'] = (adc_left_baseline + adc_left_high_quality) / 2
        features['pf_right_avg'] = (pf_right_baseline + pf_right_high_quality) / 2
        
        # Differences
        features['pd_right_diff'] = pd_right_high_quality - pd_right_baseline
        
        # Ratios
        features['pd_right_ratio'] = pd_right_high_quality / pd_right_baseline
        
        # Weighted averages
        features['adc_right_weighted'] = 0.5 * adc_right_baseline + 0.5 * adc_right_high_quality
        features['adc_left_weighted'] = 0.5 * adc_left_baseline + 0.5 * adc_left_high_quality
        features['pf_right_weighted'] = 0.3 * pf_right_baseline + 0.7 * pf_right_high_quality
        
        # Add the high-quality measurements directly
        features['adc_right_high_quality'] = adc_right_high_quality
        features['pf_right_high_quality'] = pf_right_high_quality
        
        # Extract features in the correct order
        feature_array = np.array([features[feature] for feature in top_features]).reshape(1, -1)
        
        # Standardize the features
        scaled_features = scaler.transform(feature_array)
        
        # Make prediction
        prediction = model.predict(scaled_features)[0]
        probabilities = model.predict_proba(scaled_features)[0]
        
        # Display results
        st.subheader("Prediction Results")
        
        # Use a success/error box based on prediction
        if prediction == 1:  # Healthy
            st.success(f"Prediction: Healthy (PMRI)")
        else:  # Diseased
            st.error(f"Prediction: Impaired Blood Microcirculation (PAD)")
        
        # Show confidence with a progress bar
        confidence = max(probabilities) * 100
        st.write(f"Confidence: {confidence:.2f}%")
        st.progress(int(confidence))
        
        # Show detailed probabilities
        st.write(f"Probability - Healthy: {probabilities[1]*100:.2f}%")
        st.write(f"Probability - Diseased: {probabilities[0]*100:.2f}%")
        
        # Display feature values (optional, for transparency)
        with st.expander("See derived features used for prediction"):
            for feature in top_features:
                st.write(f"{feature}: {features.get(feature, 'N/A'):.4f}")
    
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        st.write("Please make sure the model files are available and correctly named.")

# Add information about the model
st.sidebar.header("About the Model")
st.sidebar.write("""
This classification tool uses a K-Nearest Neighbors algorithm trained to distinguish between:
- Healthy volunteers (PMRI dataset)
- Patients with impaired blood microcirculation (PAD dataset)

The model achieved 79.17% accuracy in cross-validation.
""")

# Add reference information
st.sidebar.header("References")
st.sidebar.write("""
Key parameters:
- ADC: Apparent Diffusion Coefficient
- Pf: Perfusion Fraction
- Pd: Diffusion Coefficient

Normal ranges are provided as default values.
""")