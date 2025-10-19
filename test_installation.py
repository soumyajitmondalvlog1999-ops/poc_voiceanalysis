#!/usr/bin/env python3
"""
Test script to verify all dependencies are properly installed
and the voice comparison app can run without errors.
"""

import sys
import importlib

def test_imports():
    """Test if all required packages can be imported."""
    required_packages = [
        'streamlit',
        'librosa',
        'numpy',
        'soundfile',
        'matplotlib',
        'scipy',
        'io',
        'base64'
    ]
    
    print("🔍 Testing package imports...")
    failed_imports = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError as e:
            print(f"❌ {package}: {e}")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"\n❌ Failed to import: {', '.join(failed_imports)}")
        print("Please install missing packages using: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All packages imported successfully!")
        return True

def test_audio_processing():
    """Test basic audio processing functionality."""
    try:
        import librosa
        import numpy as np
        
        print("\n🔍 Testing audio processing...")
        
        # Create a simple test signal
        sr = 22050
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration))
        test_signal = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
        
        # Test feature extraction
        rms = librosa.feature.rms(y=test_signal)[0]
        zcr = librosa.feature.zero_crossing_rate(test_signal)[0]
        spectral_centroids = librosa.feature.spectral_centroid(y=test_signal, sr=sr)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=test_signal, sr=sr)[0]
        
        print("✅ RMS calculation working")
        print("✅ Zero Crossing Rate calculation working")
        print("✅ Spectral Centroid calculation working")
        print("✅ Spectral Bandwidth calculation working")
        
        return True
        
    except Exception as e:
        print(f"❌ Audio processing test failed: {e}")
        return False

def test_streamlit_components():
    """Test Streamlit-specific components."""
    try:
        import streamlit as st
        from audiorecorder import audiorecorder
        
        print("\n🔍 Testing Streamlit components...")
        print("✅ Streamlit imported successfully")
        print("✅ streamlit-audiorecorder imported successfully")
        
        return True
        
    except ImportError as e:
        # Try alternative import name
        try:
            import streamlit as st
            from audiorecorder import audiorecorder
            
            print("\n🔍 Testing Streamlit components...")
            print("✅ Streamlit imported successfully")
            print("✅ audiorecorder imported successfully")
            
            return True
        except ImportError:
            print(f"❌ Streamlit components test failed: {e}")
            return False
    except Exception as e:
        print(f"❌ Streamlit components test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🎤 Voice Comparison App - Installation Test")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 3
    
    if test_imports():
        tests_passed += 1
    
    if test_audio_processing():
        tests_passed += 1
    
    if test_streamlit_components():
        tests_passed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! The app should run successfully.")
        print("\nTo run the app, use: streamlit run app.py")
    else:
        print("❌ Some tests failed. Please check the error messages above.")
        print("Make sure to install all dependencies: pip install -r requirements.txt")
    
    return tests_passed == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
