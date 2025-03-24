# Classification Project

## Overview
This tool classifies patients as either healthy (PMRI) or having impaired blood microcirculation (PAD) based on Diffusion and Perfusion MRI parameters. The classification system uses a K-Nearest Neighbors algorithm that achieved 79.17% accuracy in Leave-One-Out Cross-Validation.

## Prerequisites
- Python 3.8 or higher
- Required packages: streamlit, pandas, numpy, scikit-learn, joblib

## Installation
1. Clone this repository:
   ```
   git clone https://github.com/yourusername/pad-pmri-classification.git
   cd pad-pmri-classification
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Make sure you have the following model files in the project directory:
   - `knn_model.joblib`
   - `scaler.joblib`
   - `top_features.joblib`

## Usage
### Running the Web Application
1. Start the Streamlit application:
   ```
   streamlit run app.py
   ```

2. Open your web browser and navigate to the URL shown in the terminal 

3. Enter the required measurements:
   - ADC Right (baseline and high-quality)
   - ADC Left (baseline and high-quality)
   - Pf Right (baseline and high-quality)
   - Pd Right (baseline and high-quality)

4. Click the "Predict" button to see the classification result

## Model Information
- Algorithm: K-Nearest Neighbors (k=3)
- Accuracy: 79.17%
- Input Features: 8 raw measurements
- Derived Features: 10 engineered features

## Key Parameters
- **ADC**: Apparent Diffusion Coefficient
- **Pf**: Perfusion Fraction
- **Pd**: Diffusion Coefficient

## Limitations
- The model was trained on a small dataset (15 healthy patients, 9 diseased patients)
- Performance may vary when applied to patients with different demographic characteristics
- The tool should be used as a decision support system and not as a replacement for clinical judgment
