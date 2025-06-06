import numpy as np
import soundfile as sf
from scipy import signal
import librosa

def extract_vocals(input_file, output_file):
    try:
        # Read the audio file
        audio_data, sample_rate = sf.read(input_file)
        
        # Convert stereo to mono if needed
        if len(audio_data.shape) == 2:
            # Split the audio into left and right channels
            left_channel = audio_data[:, 0]
            right_channel = audio_data[:, 1]
            
            # Convert to mono for processing
            audio_mono = (left_channel + right_channel) / 2
            
            # Compute STFT with smaller window size
            S = librosa.stft(audio_mono, n_fft=1024, hop_length=256)
            mag = np.abs(S)
            phase = np.angle(S)
            
            # Lighter median filtering
            filter_size = 31  # Reduced filter size
            background = signal.medfilt2d(mag, [filter_size, 1])
            
            # Simpler soft mask
            mask = np.maximum(0, mag - background)
            mask = mask / (mag + 1e-10)
            
            # Apply mask and reconstruct
            S_vocals = mask * mag * np.exp(1j * phase)
            vocals = librosa.istft(S_vocals, hop_length=256)
            
            # Basic frequency filtering
            freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=1024)
            vocal_range_mask = (freqs > 200) & (freqs < 8000)
            S_vocals_filtered = S_vocals * vocal_range_mask[:, np.newaxis]
            vocals = librosa.istft(S_vocals_filtered, hop_length=256)
            
            # Simple normalization
            vocals_final = vocals / np.max(np.abs(vocals))
            
            # Save the vocals version
            sf.write(output_file, vocals_final, sample_rate)
            print("Vocal extraction completed successfully!")
            return True
            
        else:
            raise ValueError("Input audio must be stereo (2 channels)")
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return False

if __name__ == "__main__":
    input_file = "/Users/anikethhebbar/Desktop/workspace/MTech/HPCS/OpenMP/sari_itt/cancer-cell-detection/Singular Value Decomposition (SVD) Problem  Full Explanation.mp3"
    output_file = "output_vocals.wav"
    extract_vocals(input_file, output_file)
