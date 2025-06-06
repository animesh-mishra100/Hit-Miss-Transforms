import numpy as np
import soundfile as sf
from scipy import signal
from librosa import stft, istft
import librosa

def extract_vocals(input_file, output_file):
    # Read the audio file
    audio_data, sample_rate = sf.read(input_file)
    
    # Convert stereo to mono if needed
    if len(audio_data.shape) == 2:
        # Split the audio into left and right channels
        left_channel = audio_data[:, 0]
        right_channel = audio_data[:, 1]
        
        # Convert to mono for processing
        audio_mono = (left_channel + right_channel) / 2
        
        # Compute STFT
        S = librosa.stft(audio_mono)
        mag = np.abs(S)
        phase = np.angle(S)
        
        # Median filtering for background music reduction
        filter_size = 31
        background = signal.medfilt2d(mag, [filter_size, 1])
        
        # Compute soft mask
        mask = np.maximum(0, mag - background)
        mask = mask / (mag + 1e-10)
        
        # Apply mask and reconstruct
        S_vocals = mask * mag * np.exp(1j * phase)
        vocals = librosa.istft(S_vocals)
        
        # Additional processing
        # Apply psychoacoustic filtering
        freqs = librosa.fft_frequencies(sr=sample_rate)
        vocal_range_mask = (freqs > 200) & (freqs < 8000)
        S_vocals_filtered = S_vocals * vocal_range_mask[:, np.newaxis]
        vocals = librora.istft(S_vocals_filtered)
        
        # Harmonic-percussive separation
        vocals_harmonic = librosa.effects.harmonic(vocals)
        
        # Normalize the output
        vocals = vocals_harmonic / np.max(np.abs(vocals_harmonic))
        
    else:
        raise ValueError("Input audio must be stereo (2 channels)")
    
    # Save the vocals version
    sf.write(output_file, vocals, sample_rate)

if __name__ == "__main__":
    input_file = "/Users/anikethhebbar/Desktop/workspace/MTech/HPCS/OpenMP/sari_itt/cancer-cell-detection/Singular Value Decomposition (SVD) Problem  Full Explanation.mp3"  # Replace with your input file
    output_file = "output_vocals.wav"
    extract_vocals(input_file, output_file)
