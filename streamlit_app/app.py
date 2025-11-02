import streamlit as st
from model_helper import predict

# Page config
st.set_page_config(
    page_title="Vehicle Damage Detection",
    page_icon="🚗",
    layout="centered"
)

# Custom CSS for cleaner look
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .stTitle {
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        color: #1f2937 !important;
        margin-bottom: 0.5rem !important;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #6b7280;
        margin-bottom: 2rem;
    }
    .result-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-top: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .result-label {
        font-size: 0.9rem;
        opacity: 0.9;
        margin-bottom: 0.5rem;
    }
    .result-value {
        font-size: 1.8rem;
        font-weight: 700;
    }
    .damage-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        margin-top: 0.5rem;
    }
    .damage-front { background-color: #ef4444; }
    .damage-rear { background-color: #f59e0b; }
    .damage-normal { background-color: #10b981; }
    div[data-testid="stFileUploader"] {
        border: 2px dashed #cbd5e1;
        border-radius: 12px;
        padding: 2rem;
        background-color: #f8fafc;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.title("🚗 Vehicle Damage Detection")
st.markdown('<p class="subtitle">Upload a vehicle image to detect damage location and severity</p>',
            unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader(
    'Choose an image',
    type=['jpg', 'jpeg', 'png'],
    help="Upload a clear image of the vehicle"
)

if uploaded_file:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("#### 📸 Uploaded Image")
        st.image(uploaded_file, use_container_width=True)

    with col2:
        st.markdown("#### 🔍 Analysis")
        with st.spinner('Analyzing image...'):
            # Save and predict
            image_path = 'temp_file.jpg'
            with open(image_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())

            prediction = predict(image_path)

            # Parse prediction
            location = "Front" if prediction.startswith('F_') else "Rear"
            damage_type = prediction.split('_')[1]

            # Determine badge color
            if damage_type == "Normal":
                badge_class = "damage-normal"
                status_emoji = "✅"
            elif damage_type == "Breakage":
                badge_class = "damage-front"
                status_emoji = "⚠️"
            else:  # Crushed
                badge_class = "damage-rear"
                status_emoji = "🚨"

            # Display result card
            st.markdown(f"""
                <div class="result-card">
                    <div class="result-label">Detection Result</div>
                    <div class="result-value">{status_emoji} {damage_type}</div>
                    <div class="damage-badge {badge_class}">
                        {location} Section
                    </div>
                </div>
            """, unsafe_allow_html=True)

            # Additional info
            st.markdown("---")
            st.markdown("**Classification Details:**")
            st.markdown(f"- **Location:** {location}")
            st.markdown(f"- **Damage Type:** {damage_type}")
            st.markdown(f"- **Full Class:** `{prediction}`")

else:
    # Empty state
    st.info("👆 Upload an image to get started")

    # Sample info
    with st.expander("ℹ️ About the Model"):
        st.markdown("""
        This model can detect:
        - **Front Breakage** - Minor damage to front section
        - **Front Crushed** - Severe damage to front section
        - **Front Normal** - No damage detected in front
        - **Rear Breakage** - Minor damage to rear section
        - **Rear Crushed** - Severe damage to rear section
        - **Rear Normal** - No damage detected in rear

        Built with ResNet50 and PyTorch 🔥
        """)