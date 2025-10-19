"""
Voice Comparison App
A Streamlit application for comparing voice characteristics using audio feature extraction.
"""

import streamlit as st
import librosa
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import matplotlib
from io import BytesIO
import base64
# Note: Using Streamlit's built-in audio recording instead of streamlit-audiorecorder
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Configure matplotlib for dark theme
matplotlib.use('Agg')
plt.style.use('dark_background')

# Page configuration
st.set_page_config(
    page_title="Voice Comparison App",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for dark theme
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4, #45b7d1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        margin: 1.5rem 0 1rem 0;
        color: #4ecdc4;
    }

    .metric-container {
        background-color: #1e1e1e;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #333;
        margin: 1rem 0;
    }

    .similarity-score {
        font-size: 2rem;
        font-weight: bold;
        text-align: center;
        color: #4ecdc4;
        margin: 1rem 0;
    }

    .equation-box {
        background-color: #2d2d2d;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #4ecdc4;
        margin: 1rem 0;
    }

    .stButton > button {
        background: linear-gradient(90deg, #4ecdc4, #45b7d1);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }

    .stButton > button:hover {
        background: linear-gradient(90deg, #45b7d1, #4ecdc4);
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
    }
</style>
""", unsafe_allow_html=True)


def extract_audio_features(audio_data, sr):
    """
    Extract professional audio features for voice comparison.

    Args:
        audio_data: Audio signal array
        sr: Sample rate

    Returns:
        dict: Dictionary containing extracted features
    """
    try:
        # Ensure audio is mono
        if len(audio_data.shape) > 1:
            audio_data = librosa.to_mono(audio_data)

        # Extract core voice features
        rms = np.mean(librosa.feature.rms(y=audio_data)[0])
        zcr = np.mean(librosa.feature.zero_crossing_rate(audio_data)[0])
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio_data, sr=sr)[0])
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio_data, sr=sr)[0])

        # Extract MFCC features (most important for voice)
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=12)
        mfcc_mean = np.mean(mfccs, axis=1)

        # Extract chroma features (pitch characteristics)
        chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)

        # Extract spectral rolloff
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio_data, sr=sr)[0])

        # Professional feature set
        features = {
            'RMS_Energy': rms,
            'Zero_Crossing_Rate': zcr,
            'Spectral_Centroid': spectral_centroid / 1000,  # Normalize to kHz
            'Spectral_Bandwidth': spectral_bandwidth / 1000,
            'Spectral_Rolloff': spectral_rolloff / 1000,
            'MFCC_1': mfcc_mean[0],
            'MFCC_2': mfcc_mean[1],
            'MFCC_3': mfcc_mean[2],
            'MFCC_4': mfcc_mean[3],
            'MFCC_5': mfcc_mean[4],
            'MFCC_6': mfcc_mean[5],
            'Chroma_C': chroma_mean[0],
            'Chroma_D': chroma_mean[2],
            'Chroma_E': chroma_mean[4],
            'Chroma_F': chroma_mean[5],
            'Chroma_G': chroma_mean[7],
            'Chroma_A': chroma_mean[9],
            'Chroma_B': chroma_mean[11]
        }

        return features
    except Exception as e:
        st.error(f"Error extracting features: {str(e)}")
        return None


def calculate_cosine_similarity(features1, features2):
    """
    Calculate strict voice similarity - different people should have LOW scores.

    Args:
        features1: Dictionary of features from first audio
        features2: Dictionary of features from second audio

    Returns:
        float: Cosine similarity score (0 to 1)
    """
    try:
        # Convert feature dictionaries to numpy arrays
        vec1 = np.array(list(features1.values()))
        vec2 = np.array(list(features2.values()))

        # Calculate raw cosine similarity first
        vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-8)
        vec2_norm = vec2 / (np.linalg.norm(vec2) + 1e-8)
        raw_similarity = np.dot(vec1_norm, vec2_norm)

        # Apply STRICT scaling - make it much harder to get high scores
        if raw_similarity > 0.95:
            # Only extremely similar voices get high scores
            similarity = 0.8 + (raw_similarity - 0.95) * 4.0
        elif raw_similarity > 0.90:
            # Very similar voices get moderate-high scores
            similarity = 0.6 + (raw_similarity - 0.90) * 4.0
        elif raw_similarity > 0.80:
            # Moderately similar voices get low-moderate scores
            similarity = 0.3 + (raw_similarity - 0.80) * 3.0
        elif raw_similarity > 0.60:
            # Somewhat similar voices get low scores
            similarity = 0.1 + (raw_similarity - 0.60) * 1.0
        else:
            # Different voices get very low scores
            similarity = raw_similarity * 0.5

        # Ensure result is between 0 and 1
        similarity = max(0.0, min(1.0, similarity))

        return similarity
    except Exception as e:
        st.error(f"Error calculating similarity: {str(e)}")
        return 0.0


def get_similarity_interpretation(similarity):
    """
    Get realistic interpretation of similarity score.

    Args:
        similarity: Cosine similarity score (0 to 1)

    Returns:
        tuple: (interpretation, color, threshold_info)
    """
    if similarity >= 0.75:
        return "Very High Match", "#00ff00", "Very high confidence - likely same speaker"
    elif similarity >= 0.60:
        return "High Match", "#90ee90", "High confidence - probably same speaker"
    elif similarity >= 0.40:
        return "Moderate Match", "#ffff00", "Moderate confidence - possibly same speaker"
    elif similarity >= 0.25:
        return "Low Match", "#ffa500", "Low confidence - likely different speakers"
    else:
        return "No Match", "#ff0000", "Very low confidence - definitely different speakers"


def create_comparison_chart(features1, features2, labels1="Reference", labels2="Recorded"):
    """
    Create a clean, professional bar chart comparing key voice features.

    Args:
        features1: Dictionary of features from first audio
        features2: Dictionary of features from second audio
        labels1: Label for first audio
        labels2: Label for second audio

    Returns:
        matplotlib.figure.Figure: The created figure
    """
    # Select only the most important features for display
    key_features = [
        'RMS_Energy',
        'Zero_Crossing_Rate',
        'Spectral_Centroid',
        'Spectral_Bandwidth',
        'MFCC_1',
        'MFCC_2',
        'Chroma_C',
        'Chroma_E'
    ]

    # Filter features and create clean labels
    filtered_features = []
    clean_labels = []
    values1 = []
    values2 = []

    for feature in key_features:
        if feature in features1 and feature in features2:
            filtered_features.append(feature)
            # Create clean, readable labels
            clean_label = feature.replace('_', ' ').title()
            if 'Mfcc' in clean_label:
                clean_label = clean_label.replace('Mfcc', 'MFCC')
            clean_labels.append(clean_label)
            values1.append(features1[feature])
            values2.append(features2[feature])

    fig, ax = plt.subplots(figsize=(14, 8))

    x = np.arange(len(filtered_features))
    width = 0.35

    # Create bars with professional colors
    bars1 = ax.bar(x - width / 2, values1, width, label=labels1,
                   color='#2E86AB', alpha=0.8, edgecolor='white', linewidth=1)
    bars2 = ax.bar(x + width / 2, values2, width, label=labels2,
                   color='#A23B72', alpha=0.8, edgecolor='white', linewidth=1)

    # Styling
    ax.set_xlabel('Voice Features', fontsize=14, color='white', fontweight='bold')
    ax.set_ylabel('Feature Values', fontsize=14, color='white', fontweight='bold')
    ax.set_title('Voice Feature Comparison', fontsize=18, color='white', fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(clean_labels, rotation=45, ha='right', fontsize=12, color='white')
    ax.legend(fontsize=12, framealpha=0.9)
    ax.grid(True, alpha=0.2, linestyle='--')

    # Add value labels on bars (only if values are significant)
    for bar in bars1:
        height = bar.get_height()
        if height > 0.01:  # Only show labels for significant values
            ax.text(bar.get_x() + bar.get_width() / 2., height + height * 0.01,
                    f'{height:.3f}', ha='center', va='bottom',
                    color='white', fontsize=10, fontweight='bold')

    for bar in bars2:
        height = bar.get_height()
        if height > 0.01:  # Only show labels for significant values
            ax.text(bar.get_x() + bar.get_width() / 2., height + height * 0.01,
                    f'{height:.3f}', ha='center', va='bottom',
                    color='white', fontsize=10, fontweight='bold')

    # Set background color
    ax.set_facecolor('#1a1a1a')
    fig.patch.set_facecolor('#1a1a1a')

    # Adjust layout
    plt.tight_layout()
    return fig


def create_audio_visualization(audio_data, sr, title="Audio Visualization"):
    """
    Create clean waveform and spectrogram visualization.

    Args:
        audio_data: Audio signal array
        sr: Sample rate
        title: Title for the plot

    Returns:
        matplotlib.figure.Figure: The created figure
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Waveform
    time = np.linspace(0, len(audio_data) / sr, len(audio_data))
    ax1.plot(time, audio_data, color='#2E86AB', linewidth=1.2)
    ax1.set_title(f'{title} - Waveform', color='white', fontweight='bold', fontsize=14)
    ax1.set_xlabel('Time (s)', color='white', fontsize=12)
    ax1.set_ylabel('Amplitude', color='white', fontsize=12)
    ax1.grid(True, alpha=0.2, linestyle='--')
    ax1.set_facecolor('#1a1a1a')

    # Spectrogram
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio_data)), ref=np.max)
    img = librosa.display.specshow(D, y_axis='hz', x_axis='time', sr=sr, ax=ax2,
                                   cmap='viridis', fmax=8000)
    ax2.set_title(f'{title} - Spectrogram', color='white', fontweight='bold', fontsize=14)
    ax2.set_xlabel('Time (s)', color='white', fontsize=12)
    ax2.set_ylabel('Frequency (Hz)', color='white', fontsize=12)
    ax2.set_facecolor('#1a1a1a')

    # Add colorbar
    cbar = plt.colorbar(img, ax=ax2, format='%+2.0f dB')
    cbar.ax.tick_params(colors='white')
    cbar.set_label('dB', color='white', fontsize=12)

    # Set figure background
    fig.patch.set_facecolor('#1a1a1a')

    plt.tight_layout()
    return fig


def get_download_link(fig, filename="comparison_chart.png"):
    """
    Generate a download link for matplotlib figure.

    Args:
        fig: matplotlib figure
        filename: name of the file to download

    Returns:
        str: HTML download link
    """
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight', facecolor='black')
    buf.seek(0)
    data = buf.getvalue()
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:image/png;base64,{b64}" download="{filename}">Download Chart as PNG</a>'
    return href


def main():
    """Main application function."""

    # Header
    st.markdown('<h1 class="main-header">Voice Comparison App</h1>', unsafe_allow_html=True)

    # Initialize session state
    if 'reference_audio' not in st.session_state:
        st.session_state.reference_audio = None
    if 'reference_sr' not in st.session_state:
        st.session_state.reference_sr = None
    if 'recorded_audio' not in st.session_state:
        st.session_state.recorded_audio = None
    if 'recorded_sr' not in st.session_state:
        st.session_state.recorded_sr = None

    # Create two columns for upload and record sections
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<h2 class="section-header">Upload Reference Voice</h2>', unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Choose a reference audio file",
            type=['wav', 'mp3'],
            help="Upload a .wav or .mp3 file (max 10MB)"
        )

        if uploaded_file is not None:
            # Check file size
            if uploaded_file.size > 10 * 1024 * 1024:  # 10MB
                st.error("File size too large! Please upload a file smaller than 10MB.")
            else:
                try:
                    # Load audio file
                    audio_data, sr = librosa.load(uploaded_file, sr=None)

                    # Normalize sample rate to 22050 Hz for consistency
                    if sr != 22050:
                        audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=22050)
                        sr = 22050

                    # Limit to 5 seconds
                    max_samples = 5 * sr
                    if len(audio_data) > max_samples:
                        audio_data = audio_data[:max_samples]
                        st.warning("Audio truncated to 5 seconds for processing.")

                    st.session_state.reference_audio = audio_data
                    st.session_state.reference_sr = sr

                    st.success(f"Reference audio loaded successfully!")
                    st.info(f"Duration: {len(audio_data) / sr:.2f}s, Sample Rate: {sr}Hz")

                    # Show audio player
                    st.audio(uploaded_file, format='audio/wav')

                except Exception as e:
                    st.error(f"Error loading audio file: {str(e)}")

    with col2:
        st.markdown('<h2 class="section-header">Upload Your Voice</h2>', unsafe_allow_html=True)

        # Audio file upload for recorded voice
        recorded_file = st.file_uploader(
            "Choose your recorded voice file",
            type=['wav', 'mp3'],
            help="Upload a .wav or .mp3 file of your voice (max 10MB)",
            key="recorded_upload"
        )

        if recorded_file is not None:
            # Check file size
            if recorded_file.size > 10 * 1024 * 1024:  # 10MB
                st.error("File size too large! Please upload a file smaller than 10MB.")
            else:
                try:
                    # Load audio file
                    audio_data, sr = librosa.load(recorded_file, sr=None)

                    # Normalize sample rate to 22050 Hz for consistency
                    if sr != 22050:
                        audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=22050)
                        sr = 22050

                    # Limit to 5 seconds
                    max_samples = 5 * sr
                    if len(audio_data) > max_samples:
                        audio_data = audio_data[:max_samples]
                        st.warning("Audio truncated to 5 seconds for processing.")

                    st.session_state.recorded_audio = audio_data
                    st.session_state.recorded_sr = sr

                    st.success(f"Recorded voice loaded successfully!")
                    st.info(f"Duration: {len(audio_data) / sr:.2f}s, Sample Rate: {sr}Hz")

                    # Show audio player
                    st.audio(recorded_file, format='audio/wav')

                except Exception as e:
                    st.error(f"Error loading recorded audio file: {str(e)}")

    # Analysis section
    if st.session_state.reference_audio is not None and st.session_state.recorded_audio is not None:
        st.markdown('<h2 class="section-header">Analyze Audio</h2>', unsafe_allow_html=True)

        if st.button("Analyze Voice Similarity", type="primary"):
            with st.spinner("Extracting audio features..."):
                # Extract features from both audios
                ref_features = extract_audio_features(st.session_state.reference_audio, st.session_state.reference_sr)
                rec_features = extract_audio_features(st.session_state.recorded_audio, st.session_state.recorded_sr)

                if ref_features and rec_features:
                    # Calculate similarity
                    similarity = calculate_cosine_similarity(ref_features, rec_features)

                    # Get interpretation
                    interpretation, color, threshold_info = get_similarity_interpretation(similarity)

                    # Display similarity score with interpretation
                    st.markdown(f'''
                    <div class="similarity-score" style="color: {color};">
                        Similarity Score: {similarity:.3f}
                        <br>
                        <span style="font-size: 1.2rem; font-weight: normal;">{interpretation}</span>
                    </div>
                    ''', unsafe_allow_html=True)

                    # Display threshold information
                    st.info(f"**Interpretation:** {threshold_info}")

                    # Realistic threshold reference
                    st.markdown("### Similarity Scale:")
                    col1, col2, col3, col4, col5 = st.columns(5)

                    with col1:
                        st.metric("Very High", "≥0.75", "Same speaker")
                    with col2:
                        st.metric("High", "≥0.60", "Probably same")
                    with col3:
                        st.metric("Moderate", "≥0.40", "Possibly same")
                    with col4:
                        st.metric("Low", "≥0.25", "Different speakers")
                    with col5:
                        st.metric("No Match", "<0.25", "Different speakers")

                    # Create comparison chart
                    st.markdown("### Feature Comparison Chart")
                    fig = create_comparison_chart(ref_features, rec_features, "Reference", "Recorded")
                    st.pyplot(fig)

                    # Download link
                    st.markdown(get_download_link(fig), unsafe_allow_html=True)

                    # Display key feature values
                    st.markdown("### Key Features Comparison")

                    # Show only the most important features
                    key_features = ['RMS_Energy', 'Zero_Crossing_Rate', 'Spectral_Centroid', 'MFCC_1', 'MFCC_2',
                                    'Chroma_C']

                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("**Reference Audio:**")
                        for feature in key_features:
                            if feature in ref_features:
                                st.metric(feature.replace('_', ' '), f"{ref_features[feature]:.3f}")

                    with col2:
                        st.markdown("**Recorded Audio:**")
                        for feature in key_features:
                            if feature in rec_features:
                                st.metric(feature.replace('_', ' '), f"{rec_features[feature]:.3f}")

                    # Audio visualizations
                    st.markdown("### Audio Visualizations")

                    viz_col1, viz_col2 = st.columns(2)

                    with viz_col1:
                        st.markdown("**Reference Audio:**")
                        ref_viz = create_audio_visualization(
                            st.session_state.reference_audio,
                            st.session_state.reference_sr,
                            "Reference"
                        )
                        st.pyplot(ref_viz)

                    with viz_col2:
                        st.markdown("**Recorded Audio:**")
                        rec_viz = create_audio_visualization(
                            st.session_state.recorded_audio,
                            st.session_state.recorded_sr,
                            "Recorded"
                        )
                        st.pyplot(rec_viz)

                    # Professional summary
                    st.markdown("### Analysis Summary")
                    st.markdown("""
                    <div class="equation-box">
                    <h4>Voice Comparison Analysis:</h4>
                    <p><strong>RMS Energy:</strong> Measures overall audio power and loudness</p>
                    <p><strong>Zero Crossing Rate:</strong> Indicates voice pitch and frequency content</p>
                    <p><strong>Spectral Centroid:</strong> Represents the "brightness" of the voice</p>
                    <p><strong>MFCC Coefficients:</strong> Capture vocal tract characteristics and timbre</p>
                    <p><strong>Chroma Features:</strong> Analyze pitch class distribution and musical content</p>
                    </div>
                    """, unsafe_allow_html=True)

    # Reset button
    st.markdown("---")
    if st.button("Reset Application", type="secondary"):
        for key in ['reference_audio', 'reference_sr', 'recorded_audio', 'recorded_sr']:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        <p>Voice Comparison App | Built with Streamlit & Librosa</p>
        <p>Processes audio in-memory • No external database required</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()