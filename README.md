# 🎤 Voice Comparison App

A complete Streamlit web application that allows users to upload a reference voice file and record their own voice directly in the browser. The app compares both voices and displays key audio analytics graphs and similarity scores.

## ✨ Features

- **Voice Upload**: Upload reference audio files (.wav or .mp3, max 100KB)
- **Voice Recording**: Record your voice directly in the browser using microphone
- **Audio Analysis**: Extract and compare audio features (RMS, ZCR, Spectral Centroid, Spectral Bandwidth)
- **Similarity Scoring**: Calculate cosine similarity between voice features
- **Visualizations**: Interactive bar charts, waveforms, and spectrograms
- **Download Results**: Export comparison charts as PNG files
- **Dark Theme**: Modern, clean UI with dark theme styling
- **Offline Processing**: All audio processing happens in-memory, no external database required

## 🚀 Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Application**:
   ```bash
   streamlit run app.py
   ```

3. **Open in Browser**:
   The app will automatically open at `http://localhost:8501`

## 📋 Usage Instructions

1. **Upload Reference Voice**: Use the file uploader to select a .wav or .mp3 file (max 100KB)
2. **Record Your Voice**: Click the microphone button to start recording
3. **Analyze Audio**: Click "Analyze Audio" to compare the voices
4. **View Results**: See similarity scores, feature comparisons, and visualizations
5. **Download Charts**: Save comparison graphs as PNG files

## 🔧 Technical Details

### Audio Features Extracted

- **Root Mean Square (RMS)**: Measures the average power of the audio signal
- **Zero Crossing Rate (ZCR)**: Indicates the rate of sign changes in the signal
- **Spectral Centroid**: Represents the "center of mass" of the spectrum
- **Spectral Bandwidth**: Measures the spread of the spectrum around the centroid

### Similarity Calculation

The app uses cosine similarity to compare feature vectors:
```
similarity = (A · B) / (||A|| × ||B||)
```

### Dependencies

- `streamlit`: Web application framework
- `librosa`: Audio analysis library
- `numpy`: Numerical computing
- `soundfile`: Audio file I/O
- `matplotlib`: Plotting and visualization
- `streamlit-audiorecorder`: Voice recording component
- `scipy`: Scientific computing

## 📁 Project Structure

```
voice_comparison_app/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── assets/             # Optional assets folder
└── README.md          # This file
```

## 🎯 Example Output

- **Similarity Score**: 0.8542 (High Similarity)
- **Feature Comparison Chart**: Side-by-side bar chart showing RMS, ZCR, Centroid, and Bandwidth
- **Waveform Visualization**: Time-domain representation of audio signals
- **Spectrogram**: Frequency-domain visualization of audio content

## 🔍 Troubleshooting

- **File Size Error**: Ensure uploaded files are under 100KB
- **Audio Loading Error**: Check that files are valid .wav or .mp3 format
- **Recording Issues**: Ensure microphone permissions are granted
- **Memory Issues**: The app processes audio in-memory; very large files may cause issues

## 📊 Mathematical Formulas

The app displays LaTeX equations for each audio feature:

- **RMS**: RMS = √((1/N) × Σ(y_i²))
- **ZCR**: ZCR = (1/(N-1)) × Σ[sign(y_i) ≠ sign(y_{i+1})]
- **Spectral Centroid**: C = (Σ f_k × S(k)) / (Σ S(k))
- **Spectral Bandwidth**: B = √(Σ(f_k - C)² × S(k))

## 🎨 UI Features

- Modern dark theme with gradient accents
- Responsive design with two-column layout
- Progress indicators and success notifications
- Expandable sections for detailed visualizations
- Download functionality for results
- Reset button to clear all data

## 🔒 Privacy & Security

- All audio processing happens locally in your browser
- No data is sent to external servers
- No database storage required
- Audio files are processed in-memory only

---

Built with ❤️ using Streamlit and Librosa

