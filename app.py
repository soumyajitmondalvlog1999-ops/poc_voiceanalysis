import streamlit as st
import librosa
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import matplotlib
from io import BytesIO
import base64
from audiorecorder import audiorecorder

import scipy.signal
from scipy.spatial.distance import cosine

# Set matplotlib backend to avoid GUI issues
matplotlib.use('Agg')

# Configure Streamlit page
st.set_page_config(
    page_title="🎤 Voice Comparison App",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme and modern styling
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    .uploadedFile {
        background-color: #262730;
        border: 1px solid #4a4a4a;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .metric-container {
        background-color: #262730;
        border: 1px solid #4a4a4a;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .similarity-score {
        font-size: 2rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        margin: 1rem 0;
        color: #4ecdc4;
    }
    .info-box {
        background-color: #1e3a8a;
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.25rem;
    }
    .success-box {
        background-color: #064e3b;
        border-left: 4px solid #10b981;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.25rem;
    }
    .warning-box {
        background-color: #78350f;
        border-left: 4px solid #f59e0b;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'reference_audio' not in st.session_state:
    st.session_state.reference_audio = None
if 'recorded_audio' not in st.session_state:
    st.session_state.recorded_audio = None
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False

def extract_audio_features(audio_data, sr):
    """
    Extract audio features from audio data and sample rate.
    
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
        
        # Extract features
        rms = librosa.feature.rms(y=audio_data)[0]
        zcr = librosa.feature.zero_crossing_rate(audio_data)[0]
        spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sr)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=sr)[0]
        
        # Calculate mean values
        features = {
            'RMS': np.mean(rms),
            'ZCR': np.mean(zcr),
            'Spectral Centroid': np.mean(spectral_centroids),
            'Spectral Bandwidth': np.mean(spectral_bandwidth)
        }
        
        return features
    except Exception as e:
        st.error(f"Error extracting features: {str(e)}")
        return None

def calculate_cosine_similarity(features1, features2):
    """
    Calculate cosine similarity between two feature vectors.
    
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
        
        # Calculate cosine similarity
        cosine_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        
        # Ensure similarity is between 0 and 1
        cosine_sim = max(0, min(1, cosine_sim))
        
        return cosine_sim
    except Exception as e:
        st.error(f"Error calculating similarity: {str(e)}")
        return 0.0

def create_comparison_chart(features1, features2, title1="Reference", title2="Recorded"):
    """
    Create a bar chart comparing features between two audio files.
    
    Args:
        features1: Dictionary of features from first audio
        features2: Dictionary of features from second audio
        title1: Title for first audio
        title2: Title for second audio
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    try:
        # Set up the plot
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.patch.set_facecolor('#0e1117')
        ax.set_facecolor('#262730')
        
        # Prepare data
        features = list(features1.keys())
        values1 = list(features1.values())
        values2 = list(features2.values())
        
        # Set up bar positions
        x = np.arange(len(features))
        width = 0.35
        
        # Create bars
        bars1 = ax.bar(x - width/2, values1, width, label=title1, color='#4ecdc4', alpha=0.8)
        bars2 = ax.bar(x + width/2, values2, width, label=title2, color='#ff6b6b', alpha=0.8)
        
        # Customize the plot
        ax.set_xlabel('Audio Features', color='#fafafa', fontsize=12)
        ax.set_ylabel('Feature Values', color='#fafafa', fontsize=12)
        ax.set_title('Audio Feature Comparison', color='#fafafa', fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(features, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Color the axes and labels
        ax.tick_params(colors='#fafafa')
        ax.spines['bottom'].set_color('#fafafa')
        ax.spines['left'].set_color('#fafafa')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}', ha='center', va='bottom', color='#fafafa', fontsize=10)
        
        for bar in bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}', ha='center', va='bottom', color='#fafafa', fontsize=10)
        
        plt.tight_layout()
        return fig
    except Exception as e:
        st.error(f"Error creating comparison chart: {str(e)}")
        return None

def create_waveform_plot(audio_data, sr, title="Waveform"):
    """
    Create a waveform visualization.
    
    Args:
        audio_data: Audio signal array
        sr: Sample rate
        title: Title for the plot
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    try:
        fig, ax = plt.subplots(figsize=(10, 4))
        fig.patch.set_facecolor('#0e1117')
        ax.set_facecolor('#262730')
        
        # Create time axis
        time = np.linspace(0, len(audio_data) / sr, len(audio_data))
        
        # Plot waveform
        ax.plot(time, audio_data, color='#4ecdc4', linewidth=0.8)
        ax.set_xlabel('Time (seconds)', color='#fafafa')
        ax.set_ylabel('Amplitude', color='#fafafa')
        ax.set_title(title, color='#fafafa', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Color the axes
        ax.tick_params(colors='#fafafa')
        ax.spines['bottom'].set_color('#fafafa')
        ax.spines['left'].set_color('#fafafa')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        return fig
    except Exception as e:
        st.error(f"Error creating waveform plot: {str(e)}")
        return None

def create_spectrogram(audio_data, sr, title="Spectrogram"):
    """
    Create a spectrogram visualization.
    
    Args:
        audio_data: Audio signal array
        sr: Sample rate
        title: Title for the plot
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    try:
        fig, ax = plt.subplots(figsize=(10, 4))
        fig.patch.set_facecolor('#0e1117')
        ax.set_facecolor('#262730')
        
        # Create spectrogram
        D = librosa.amplitude_to_db(np.abs(librosa.stft(audio_data)), ref=np.max)
        img = librosa.display.specshow(D, y_axis='hz', x_axis='time', sr=sr, ax=ax, cmap='viridis')
        
        ax.set_title(title, color='#fafafa', fontweight='bold')
        ax.set_xlabel('Time (seconds)', color='#fafafa')
        ax.set_ylabel('Frequency (Hz)', color='#fafafa')
        
        # Color the axes
        ax.tick_params(colors='#fafafa')
        ax.spines['bottom'].set_color('#fafafa')
        ax.spines['left'].set_color('#fafafa')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add colorbar
        cbar = plt.colorbar(img, ax=ax)
        cbar.set_label('dB', color='#fafafa')
        cbar.ax.tick_params(colors='#fafafa')
        
        plt.tight_layout()
        return fig
    except Exception as e:
        st.error(f"Error creating spectrogram: {str(e)}")
        return None

def get_download_link(fig, filename="comparison_chart.png"):
    """
    Generate a download link for matplotlib figure.
    
    Args:
        fig: matplotlib figure
        filename: name of the file to download
        
    Returns:
        str: HTML download link
    """
    try:
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=300, bbox_inches='tight', 
                   facecolor='#0e1117', edgecolor='none')
        buf.seek(0)
        data = buf.getvalue()
        b64 = base64.b64encode(data).decode()
        href = f'<a href="data:image/png;base64,{b64}" download="{filename}">📥 Download {filename}</a>'
        return href
    except Exception as e:
        st.error(f"Error creating download link: {str(e)}")
        return ""

# Main App
def main():
    # Header
    st.markdown("# 🎤 Voice Comparison App")
    st.markdown("---")
    
    # Info section
    st.markdown("""
    <div class="info-box">
    <strong>📋 Instructions:</strong><br>
    1. Upload a reference voice file (.wav or .mp3, max 100KB)<br>
    2. Record your voice using the microphone<br>
    3. Click "Analyze Audio" to compare the voices<br>
    4. View similarity scores and download comparison charts
    </div>
    """, unsafe_allow_html=True)
    
    # Create two columns for upload and record sections
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="section-header">📁 Upload Reference Voice</div>', unsafe_allow_html=True)
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a reference audio file",
            type=['wav', 'mp3'],
            help="Upload a .wav or .mp3 file (max 100KB)"
        )
        
        if uploaded_file is not None:
            # Check file size
            if uploaded_file.size > 100 * 1024:  # 100KB
                st.error("⚠️ File size exceeds 100KB limit. Please upload a smaller file.")
            else:
                try:
                    # Load audio file
                    audio_data, sr = librosa.load(uploaded_file, sr=None)
                    
                    # Store in session state
                    st.session_state.reference_audio = {
                        'data': audio_data,
                        'sr': sr,
                        'filename': uploaded_file.name
                    }
                    
                    st.success(f"✅ Reference audio loaded: {uploaded_file.name}")
                    st.info(f"📊 Duration: {len(audio_data)/sr:.2f} seconds, Sample Rate: {sr} Hz")
                    
                    # Show waveform
                    with st.expander("🔍 View Reference Waveform"):
                        fig_wave = create_waveform_plot(audio_data, sr, f"Reference: {uploaded_file.name}")
                        if fig_wave:
                            st.pyplot(fig_wave)
                    
                    # Show spectrogram
                    with st.expander("🌊 View Reference Spectrogram"):
                        fig_spec = create_spectrogram(audio_data, sr, f"Spectrogram: {uploaded_file.name}")
                        if fig_spec:
                            st.pyplot(fig_spec)
                            
                except Exception as e:
                    st.error(f"❌ Error loading audio file: {str(e)}")
    
    with col2:
        st.markdown('<div class="section-header">🎙️ Record Your Voice</div>', unsafe_allow_html=True)
        
        # Audio recorder
        # Audio recorder
        audio_bytes = audiorecorder()
        
        if audio_bytes is not None:
            try:
                # Load recorded audio
                audio_data, sr = sf.read(BytesIO(audio_bytes))
                
                # Convert to mono if stereo
                if len(audio_data.shape) > 1:
                    audio_data = librosa.to_mono(audio_data)
                
                # Store in session state
                st.session_state.recorded_audio = {
                    'data': audio_data,
                    'sr': sr,
                    'filename': 'recorded_voice.wav'
                }
                
                st.success("✅ Voice recording completed!")
                st.info(f"📊 Duration: {len(audio_data)/sr:.2f} seconds, Sample Rate: {sr} Hz")
                
                # Show waveform
                with st.expander("🔍 View Recorded Waveform"):
                    fig_wave = create_waveform_plot(audio_data, sr, "Recorded Voice")
                    if fig_wave:
                        st.pyplot(fig_wave)
                
                # Show spectrogram
                with st.expander("🌊 View Recorded Spectrogram"):
                    fig_spec = create_spectrogram(audio_data, sr, "Recorded Voice Spectrogram")
                    if fig_spec:
                        st.pyplot(fig_spec)
                        
            except Exception as e:
                st.error(f"❌ Error processing recorded audio: {str(e)}")
    
    # Analysis section
    if st.session_state.reference_audio is not None and st.session_state.recorded_audio is not None:
        st.markdown('<div class="section-header">🔬 Audio Analysis</div>', unsafe_allow_html=True)
        
        if st.button("🚀 Analyze Audio", type="primary", use_container_width=True):
            with st.spinner("🔄 Extracting features and analyzing audio..."):
                try:
                    # Extract features from both audios
                    ref_features = extract_audio_features(
                        st.session_state.reference_audio['data'],
                        st.session_state.reference_audio['sr']
                    )
                    
                    rec_features = extract_audio_features(
                        st.session_state.recorded_audio['data'],
                        st.session_state.recorded_audio['sr']
                    )
                    
                    if ref_features and rec_features:
                        # Calculate similarity
                        similarity = calculate_cosine_similarity(ref_features, rec_features)
                        
                        # Store results in session state
                        st.session_state.analysis_complete = True
                        st.session_state.ref_features = ref_features
                        st.session_state.rec_features = rec_features
                        st.session_state.similarity = similarity
                        
                        st.success("✅ Analysis completed successfully!")
                    else:
                        st.error("❌ Failed to extract features from audio files.")
                        
                except Exception as e:
                    st.error(f"❌ Error during analysis: {str(e)}")
    
    # Display results
    if st.session_state.analysis_complete:
        st.markdown("---")
        st.markdown('<div class="section-header">📊 Analysis Results</div>', unsafe_allow_html=True)
        
        # Similarity score
        similarity = st.session_state.similarity
        st.markdown(f"""
        <div class="similarity-score">
        🎯 Similarity Score: {similarity:.4f}
        </div>
        """, unsafe_allow_html=True)
        
        # Interpretation
        if similarity >= 0.8:
            interpretation = "🟢 Very High Similarity"
            color = "#10b981"
        elif similarity >= 0.6:
            interpretation = "🟡 High Similarity"
            color = "#f59e0b"
        elif similarity >= 0.4:
            interpretation = "🟠 Moderate Similarity"
            color = "#f97316"
        else:
            interpretation = "🔴 Low Similarity"
            color = "#ef4444"
        
        st.markdown(f"""
        <div style="text-align: center; font-size: 1.2rem; color: {color}; margin: 1rem 0;">
        {interpretation}
        </div>
        """, unsafe_allow_html=True)
        
        # Feature comparison chart
        st.markdown("### 📈 Feature Comparison")
        fig_comparison = create_comparison_chart(
            st.session_state.ref_features,
            st.session_state.rec_features,
            "Reference Voice",
            "Recorded Voice"
        )
        
        if fig_comparison:
            st.pyplot(fig_comparison)
            
            # Download button
            st.markdown("### 💾 Download Results")
            download_link = get_download_link(fig_comparison, "voice_comparison.png")
            if download_link:
                st.markdown(download_link, unsafe_allow_html=True)
        
        # Feature values table
        st.markdown("### 📋 Feature Values")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Reference Voice**")
            for feature, value in st.session_state.ref_features.items():
                st.metric(feature, f"{value:.4f}")
        
        with col2:
            st.markdown("**Recorded Voice**")
            for feature, value in st.session_state.rec_features.items():
                st.metric(feature, f"{value:.4f}")
        
        with col3:
            st.markdown("**Difference**")
            for feature in st.session_state.ref_features.keys():
                diff = abs(st.session_state.ref_features[feature] - st.session_state.rec_features[feature])
                st.metric(f"Δ {feature}", f"{diff:.4f}")
        
        # LaTeX equations
        st.markdown("### 🧮 Audio Feature Equations")
        st.markdown("""
        <div class="metric-container">
        <h4>Root Mean Square (RMS)</h4>
        <p>RMS = √((1/N) × Σ(y_i²))</p>
        
        <h4>Zero Crossing Rate (ZCR)</h4>
        <p>ZCR = (1/(N-1)) × Σ[sign(y_i) ≠ sign(y_{i+1})]</p>
        
        <h4>Spectral Centroid (C)</h4>
        <p>C = (Σ f_k × S(k)) / (Σ S(k))</p>
        
        <h4>Spectral Bandwidth (B)</h4>
        <p>B = √(Σ(f_k - C)² × S(k))</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Reset button
    if st.session_state.reference_audio is not None or st.session_state.recorded_audio is not None:
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("🔄 Reset App", use_container_width=True):
                # Clear session state
                for key in ['reference_audio', 'recorded_audio', 'analysis_complete', 
                           'ref_features', 'rec_features', 'similarity']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #6b7280; margin-top: 2rem;">
    <p>🎤 Voice Comparison App | Built with Streamlit & Librosa</p>
    <p>Processing audio in-memory • No external database required</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
