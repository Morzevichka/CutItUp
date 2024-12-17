from .media_processing.video_handler import VideoProcessing
from .media_processing.audio_handler import AudioProcessing

from config import basedir

import torch
from moviepy.video.io.VideoFileClip import VideoFileClip
from app.model.model import *
import os
import numpy as np

class VideoTrimmer:
    def __init__(self):
        self.data_path = os.path.join(basedir, 'app', 'model', 'data')
        self.chat_dir = os.path.join(self.data_path, 'videos', 'chats')
        self.models = self.load_model(os.path.join(self.data_path, 'models'), 'Net_MN_Large')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def predict_models(self, models, video, audio):
        video, audio = video.to(self.device), audio.to(self.device)

        all_predictions = []
        for model in models:
            model = model.to(self.device)
            model.eval()
            with torch.no_grad():
                pred = (model(video, audio) > 0.5).float().cpu().numpy()
            all_predictions.append(pred)
        all_predictions = np.array(all_predictions).reshape(len(models), -1)

        return np.where(np.count_nonzero(all_predictions == 1, axis=0) > 2, 1, 0)
        
    def predict_model(self, model, video, audio):
        model, video, audio = self.models.to(self.device), video.to(self.device), audio.to(self.device)
        model.eval()
        with torch.no_grad():
            pred = (model(video, audio) > 0.5).float().view(-1).cpu().numpy()
        return pred
    
    def extend_video(self, pred, frames):
        x_original = np.linspace(0, 1, len(pred))
        x_target = np.linspace(0, 1, frames)

        interpolated = np.interp(x_target, x_original, pred)
        stretched = (interpolated > 0.5).astype(int)
        return stretched
    
    def find_interesting_moments(self, video):
        changes = np.diff(np.concatenate(([0], video, [0]))) 
        starts = np.where(changes == 1)[0]
        ends = np.where(changes == -1)[0] - 1

        sequences = np.column_stack((starts, ends))
        sequences = sorted(sequences, key=lambda x: x[1] - x[0], reverse=True)
        return np.array(sequences)
    
    def trim_video(self, video_path, intervals, num_clips, target_len=20):
        intervals = intervals[:num_clips]

        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_folder = os.path.dirname(video_path)

        video = VideoFileClip(video_path)
        fps = video.fps
        len_video = video.duration

        video_names = []

        for start, end in intervals:
            start, end = start / fps, end / fps 
            duration_clip = end - start

            if duration_clip < target_len:
                stretch_left = (target_len - duration_clip) * 2 / 3
                stretch_right = (target_len - duration_clip) * 1 / 3

            if start - stretch_left >= 0:
                start -= stretch_left
            else:
                start = 0
                end = start + target_len

            if end + stretch_right <= len_video:
                end += stretch_right
            else:
                end = len_video

            duration_clip = end - start
            if duration_clip > target_len:
                end = start + target_len

            # if duration_clip < target_len:
            #     if end > len_video * 0.9:
            #         start -= (target_len - duration_clip) * 2/3
            #     elif start < len_video * 0.1:
            #         start = 0 
            #         end += (target_len - duration_clip) * 1/3
            #     else:
            #         if abs(target_len - duration_clip) * 2/3 > start:
            #             start = 0
            #         else:
            #             start -= abs(target_len - duration_clip) * 2/3
            #         end += (target_len - duration_clip) * 1/3

            print(end, start)
            trimmed = video.subclip(start, end)
            video_names.append(f'{video_name}_{start:.2f}-{end:.2f}.mp4')
            output_path = os.path.join(output_folder, f'{video_name}_{start:.2f}-{end:.2f}.mp4')
            trimmed.write_videofile(output_path, codec='libx264')
        video.close()

        return video_names
    
    def predict_trimmed(self, chat_id, filename, clips, rec_clip_duration=10):
        if chat_id is None:
            video_path = os.path.join(self.data_path, 'videos', filename)
        else:
            video_path = os.path.join(self.chat_dir, chat_id, filename)

        video, frames = VideoProcessing().get_video(video_path, height=96, width=96)
        audio = AudioProcessing().get_audio(video_path, video.shape[0])

        if isinstance(models, list):
            prediction = self.predict_models(self.models, torch.Tensor(video), torch.Tensor(audio))
        else:
            prediction = self.predict_model(self.models, torch.Tensor(video), torch.Tensor(audio))

        video_predictions = self.extend_video(prediction, frames)

        interesting_moments = self.find_interesting_moments(video_predictions)

        video_names = self.trim_video(video_path, interesting_moments, clips, target_len=rec_clip_duration)

        return video_names
    
    def load_model(self, model_path, name):
        model = Net_MN_Large().__class__()
        model.load_state_dict(torch.load(os.path.join(model_path, f'{name}.pth'), weights_only=True))
        return model