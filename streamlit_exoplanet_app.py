import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="NASA Exoplanet Classifier",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-bottom: 1rem;
        border-bottom: 2px solid #ff7f0e;
        padding-bottom: 0.5rem;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    .confirmed {
        background: linear-gradient(135deg, #d4edda, #c3e6cb);
        border: 3px solid #28a745;
        color: #155724;
    }
    .false-positive {
        background: linear-gradient(135deg, #f8d7da, #f5c6cb);
        border: 3px solid #dc3545;
        color: #721c24;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #007bff;
        margin: 0.5rem 0;
    }
    .stButton > button {
        width: 100%;
        border-radius: 10px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def create_nasa_exoplanet_dataset(n_samples=5000, random_state=42):
    """Create synthetic NASA exoplanet dataset based on Kepler mission data"""
    np.random.seed(random_state)

    # Generate realistic astronomical features
    stellar_mass = np.random.normal(1.0, 0.3, n_samples)
    stellar_radius = np.random.normal(1.0, 0.4, n_samples)
    stellar_temp = np.random.normal(5800, 800, n_samples)
    orbital_period = np.random.lognormal(2, 1.5, n_samples)
    planet_radius = np.random.lognormal(0.5, 0.8, n_samples)
    transit_depth = np.random.exponential(0.001, n_samples)
    impact_parameter = np.random.uniform(0, 1, n_samples)
    transit_duration = np.random.lognormal(1.5, 0.5, n_samples)

    # Create realistic target variable with feature correlations
    confirmation_prob = (
        (transit_depth > 0.0005) * 0.3 +
        (planet_radius > 0.8) * 0.2 +
        (orbital_period < 50) * 0.2 +
        (stellar_temp > 5000) * 0.1 +
        (stellar_temp < 7000) * 0.1 +
        np.random.normal(0, 0.1, n_samples)
    )

    koi_disposition = (confirmation_prob > 0.4).astype(int)

    return pd.DataFrame({
        'stellar_mass': stellar_mass,
        'stellar_radius': stellar_radius,
        'stellar_temperature': stellar_temp,
        'orbital_period': orbital_period,
        'planet_radius': planet_radius,
        'transit_depth': transit_depth,
        'impact_parameter': impact_parameter,
        'transit_duration': transit_duration,
        'koi_disposition': koi_disposition
    })

@st.cache_data
def remove_outliers_iqr(df, _columns):
    """Remove outliers using IQR method - optimized for Streamlit caching"""
    df_clean = df.copy()
    for col in _columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    return df_clean

@st.cache_resource
def train_model():
    """Train and cache the machine learning model"""
    # Create dataset
    data = create_nasa_exoplanet_dataset()
    features = data.drop('koi_disposition', axis=1)
    target = data['koi_disposition']

    # Preprocess data
    data_clean = remove_outliers_iqr(data, list(features.columns))
    features_clean = data_clean.drop('koi_disposition', axis=1)
    target_clean = data_clean['koi_disposition']

    # Train-test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        features_clean, target_clean, 
        test_size=0.2, 
        random_state=42, 
        stratify=target_clean
    )

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Random Forest model with optimized parameters
    model = RandomForestClassifier(
        n_estimators=100, 
        max_depth=10,
        random_state=42, 
        class_weight='balanced',
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)

    # Calculate performance metrics
    train_accuracy = model.score(X_train_scaled, y_train)
    test_accuracy = model.score(X_test_scaled, y_test)

    return {
        'model': model,
        'scaler': scaler,
        'feature_names': list(features_clean.columns),
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'dataset_size': len(data_clean),
        'original_size': len(data)
    }

def predict_exoplanet(model_data, stellar_mass, stellar_radius, stellar_temperature,
                     orbital_period, planet_radius, transit_depth,
                     impact_parameter, transit_duration):
    """Make prediction with confidence scores"""
    # Prepare input data
    input_array = np.array([[
        stellar_mass, stellar_radius, stellar_temperature,
        orbital_period, planet_radius, transit_depth,
        impact_parameter, transit_duration
    ]])

    # Scale features
    input_scaled = model_data['scaler'].transform(input_array)

    # Make prediction
    prediction = model_data['model'].predict(input_scaled)[0]
    probabilities = model_data['model'].predict_proba(input_scaled)[0]
    confidence = probabilities.max()

    result = "Confirmed Exoplanet" if prediction == 1 else "False Positive"
    return result, confidence, probabilities

def set_preset_values(preset_type):
    """Set preset parameter values for quick testing"""
    presets = {
        'earth_like': {
            'stellar_mass': 1.0, 'stellar_radius': 1.0, 'stellar_temperature': 5778,
            'orbital_period': 365.25, 'planet_radius': 1.0, 'transit_depth': 0.0008,
            'impact_parameter': 0.3, 'transit_duration': 6.5
        },
        'hot_jupiter': {
            'stellar_mass': 1.2, 'stellar_radius': 1.1, 'stellar_temperature': 6200,
            'orbital_period': 3.5, 'planet_radius': 11.2, 'transit_depth': 0.012,
            'impact_parameter': 0.1, 'transit_duration': 2.8
        }
    }
    return presets.get(preset_type, {})

def main():
    # Header section
    st.markdown('<h1 class="main-header">🌌 NASA Exoplanet Classifier</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; font-size: 1.2rem; color: #666; margin-bottom: 2rem;'>
        <strong>Advanced Machine Learning for Astronomical Data Science</strong><br>
        Classify stellar observations as confirmed exoplanets or false positives using AI
    </div>
    """, unsafe_allow_html=True)

    # Load model with progress indication
    with st.spinner("🚀 Loading AI model and preprocessing astronomical data..."):
        model_data = train_model()

    # Success message with model stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.success(f"🎯 Model Accuracy: {model_data['test_accuracy']:.1%}")
    with col2:
        st.info(f"📊 Dataset: {model_data['dataset_size']:,} samples")
    with col3:
        st.info(f"🧠 Algorithm: Random Forest")

    # Sidebar for parameters
    with st.sidebar:
        st.markdown('<h2 class="sub-header">🔭 Observation Parameters</h2>', unsafe_allow_html=True)

        # Preset buttons
        st.markdown("**🎯 Quick Test Scenarios:**")
        col1, col2 = st.columns(2)

        with col1:
            if st.button("🌍 Earth-like", help="Set parameters similar to Earth"):
                preset = set_preset_values('earth_like')
                for key, value in preset.items():
                    st.session_state[key] = value
                st.experimental_rerun()

        with col2:
            if st.button("🪐 Hot Jupiter", help="Typical gas giant parameters"):
                preset = set_preset_values('hot_jupiter')
                for key, value in preset.items():
                    st.session_state[key] = value
                st.experimental_rerun()

        st.markdown("---")
        st.markdown("**⚙️ Manual Parameter Input:**")

        # Parameter inputs with enhanced descriptions
        stellar_mass = st.slider(
            "Stellar Mass (M☉)", 0.1, 3.0, 
            st.session_state.get('stellar_mass', 1.0), 0.1,
            help="Mass of the host star relative to our Sun (1.0 = Sun's mass)"
        )

        stellar_radius = st.slider(
            "Stellar Radius (R☉)", 0.1, 3.0, 
            st.session_state.get('stellar_radius', 1.0), 0.1,
            help="Radius of the host star relative to our Sun (1.0 = Sun's radius)"
        )

        stellar_temperature = st.slider(
            "Stellar Temperature (K)", 3000, 8000, 
            st.session_state.get('stellar_temperature', 5800), 100,
            help="Effective temperature of the host star (Sun = 5,778K)"
        )

        orbital_period = st.slider(
            "Orbital Period (days)", 0.5, 500.0, 
            st.session_state.get('orbital_period', 10.0), 0.5,
            help="Time for the planet to complete one orbit around its star"
        )

        planet_radius = st.slider(
            "Planet Radius (R🌍)", 0.1, 10.0, 
            st.session_state.get('planet_radius', 1.5), 0.1,
            help="Size of the planet relative to Earth (1.0 = Earth's radius)"
        )

        transit_depth = st.slider(
            "Transit Depth", 0.0001, 0.01, 
            st.session_state.get('transit_depth', 0.001), 0.0001, "%.4f",
            help="Fraction of starlight blocked when planet passes in front of star"
        )

        impact_parameter = st.slider(
            "Impact Parameter", 0.0, 1.0, 
            st.session_state.get('impact_parameter', 0.5), 0.01,
            help="Geometric parameter: 0 = center transit, 1 = grazing transit"
        )

        transit_duration = st.slider(
            "Transit Duration (hours)", 0.5, 20.0, 
            st.session_state.get('transit_duration', 5.0), 0.5,
            help="How long the planet takes to cross in front of the star"
        )

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown('<h2 class="sub-header">🎯 AI Prediction Results</h2>', unsafe_allow_html=True)

        # Make prediction
        result, confidence, probabilities = predict_exoplanet(
            model_data, stellar_mass, stellar_radius, stellar_temperature,
            orbital_period, planet_radius, transit_depth,
            impact_parameter, transit_duration
        )

        # Enhanced prediction display
        if result == "Confirmed Exoplanet":
            st.markdown(f"""
            <div class="prediction-box confirmed">
                <h2>✅ {result}</h2>
                <h3>Confidence: {confidence:.1%}</h3>
                <p style='font-size: 1.1rem; margin-top: 1rem;'>
                    🎉 This observation shows strong evidence of a genuine exoplanet detection!
                    The AI model is highly confident in this classification.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="prediction-box false-positive">
                <h2>❌ {result}</h2>
                <h3>Confidence: {confidence:.1%}</h3>
                <p style='font-size: 1.1rem; margin-top: 1rem;'>
                    ⚠️ This observation is likely a false alarm or instrumental artifact.
                    Additional verification would be needed before confirmation.
                </p>
            </div>
            """, unsafe_allow_html=True)

        # Probability visualization
        st.markdown('<h3 class="sub-header">📊 Classification Probabilities</h3>', unsafe_allow_html=True)

        prob_df = pd.DataFrame({
            'Classification': ['False Positive', 'Confirmed Exoplanet'],
            'Probability': probabilities,
            'Percentage': [f"{p:.1%}" for p in probabilities]
        })

        fig = px.bar(
            prob_df, x='Classification', y='Probability',
            color='Classification', text='Percentage',
            color_discrete_map={'False Positive': '#ff6b6b', 'Confirmed Exoplanet': '#51cf66'},
            title="AI Model Confidence Distribution"
        )
        fig.update_traces(textposition='outside')
        fig.update_layout(showlegend=False, height=400, yaxis=dict(range=[0, 1]))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<h2 class="sub-header">📈 Feature Importance</h2>', unsafe_allow_html=True)

        # Feature importance analysis
        importance_df = pd.DataFrame({
            'Feature': model_data['feature_names'],
            'Importance': model_data['model'].feature_importances_
        }).sort_values('Importance', ascending=True)

        # Clean feature names for display
        feature_name_map = {
            'stellar_mass': 'Stellar Mass',
            'stellar_radius': 'Stellar Radius', 
            'stellar_temperature': 'Stellar Temperature',
            'orbital_period': 'Orbital Period',
            'planet_radius': 'Planet Radius',
            'transit_depth': 'Transit Depth',
            'impact_parameter': 'Impact Parameter',
            'transit_duration': 'Transit Duration'
        }
        importance_df['Feature_Clean'] = importance_df['Feature'].map(feature_name_map)

        fig_importance = px.bar(
            importance_df, x='Importance', y='Feature_Clean',
            orientation='h', color='Importance',
            color_continuous_scale='viridis',
            title="What the AI Considers Most Important"
        )
        fig_importance.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_importance, use_container_width=True)

        # Key insights
        top_feature = importance_df.iloc[-1]['Feature_Clean']
        st.markdown(f"""
        <div class="metric-card">
            <strong>🔍 Key Insight:</strong><br>
            <em>{top_feature}</em> is the most important factor for exoplanet classification.
        </div>
        """, unsafe_allow_html=True)

    # Parameter summary table
    st.markdown('<h2 class="sub-header">📋 Current Observation Summary</h2>', unsafe_allow_html=True)

    summary_data = {
        'Parameter': ['Stellar Mass', 'Stellar Radius', 'Stellar Temperature', 'Orbital Period',
                     'Planet Radius', 'Transit Depth', 'Impact Parameter', 'Transit Duration'],
        'Value': [f"{stellar_mass:.1f} M☉", f"{stellar_radius:.1f} R☉", f"{stellar_temperature:,.0f} K",
                 f"{orbital_period:.1f} days", f"{planet_radius:.1f} R🌍", f"{transit_depth:.4f}",
                 f"{impact_parameter:.2f}", f"{transit_duration:.1f} hours"],
        'Interpretation': [
            'Compared to Sun' if 0.8 <= stellar_mass <= 1.2 else 'Different from Sun',
            'Compared to Sun' if 0.8 <= stellar_radius <= 1.2 else 'Different from Sun',
            'Sun-like star' if 5000 <= stellar_temperature <= 6500 else 'Different spectral class',
            'Short period' if orbital_period < 10 else 'Long period' if orbital_period > 100 else 'Moderate period',
            'Earth-like' if 0.8 <= planet_radius <= 1.2 else 'Super-Earth' if planet_radius > 1.2 else 'Sub-Earth',
            'Deep transit' if transit_depth > 0.002 else 'Shallow transit',
            'Central transit' if impact_parameter < 0.3 else 'Grazing transit' if impact_parameter > 0.7 else 'Normal transit',
            'Long transit' if transit_duration > 8 else 'Short transit'
        ]
    }

    summary_df = pd.DataFrame(summary_data)
    st.table(summary_df)

    # Model information and help
    with st.expander("ℹ️ About This AI Model & How to Interpret Results"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"""
            **🤖 Model Specifications:**
            - **Algorithm:** Random Forest Classifier
            - **Training Accuracy:** {model_data['train_accuracy']:.1%}
            - **Test Accuracy:** {model_data['test_accuracy']:.1%}
            - **Dataset Size:** {model_data['dataset_size']:,} samples
            - **Features:** 8 astronomical parameters
            - **Classes:** Binary classification

            **🔬 Data Processing:**
            - Outlier removal using IQR method
            - Feature standardization (zero mean, unit variance)
            - Stratified train-test split (80-20)
            - Balanced class weights for handling imbalance
            """)

        with col2:
            st.markdown("""
            **🎯 Interpretation Guide:**

            **Confirmed Exoplanet Indicators:**
            - High transit depth (>0.002)
            - Reasonable planet radius (0.5-15 R🌍)
            - Low impact parameter (<0.5)
            - Consistent stellar parameters

            **False Positive Indicators:**
            - Very shallow transit depth (<0.0003)
            - Extreme impact parameter (>0.8)
            - Inconsistent parameter combinations
            - Very short or long transit durations

            **Confidence Levels:**
            - >90%: High confidence prediction
            - 70-90%: Moderate confidence
            - <70%: Low confidence, needs review
            """)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <strong>🌌 NASA Exoplanet Classifier</strong> • Built with Streamlit & Scikit-learn<br>
        <em>Advancing astronomical discovery through artificial intelligence</em>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()