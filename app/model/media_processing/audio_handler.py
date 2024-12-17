from moviepy.editor import AudioFileClip
import numpy as np

from app.model.media_processing.video_handler import VideoProcessing

class AudioProcessing:
    def get_audio(self, video_path: str, target_len: int, train=False):
        audio = AudioFileClip(video_path).to_soundarray(fps=16000)
        audio = audio.mean(axis=1)
        
        frames, fps = VideoProcessing().get_len_fps(video_path)
        
        audio = self.max_min_pooling(audio, len(audio) // frames)
        indices = np.linspace(0, len(audio) - 1, target_len).astype(int)
        audio = abs(audio[indices])
        return audio.reshape(-1, 1)
    
    def max_min_pooling(self, signal, window_size):
        windows = np.lib.stride_tricks.sliding_window_view(signal, window_size)
        max_vals = np.max(windows, axis=1)
        min_vals = np.min(windows, axis=1)
        return (max_vals + min_vals) / 2
