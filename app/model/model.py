import torch.nn as nn 
import torch
import torchvision.models as models
    
class Net_MN_Small_Lite(nn.Module):

    """
    A neural network model that combines video and audio features for classification tasks.
    
    The model uses:
    - MobileNetV3_Small as the feature extractor for video data.
    - GRU layers to process video and audio features over time.
    - CNN layers to process audio features.

    Input:
    - video (Tensor): shape (frames, 3, height, width), where 'frames' dimension is number of frames after video preprocessing
    `height` and `width` are the dimensions of the video frames (96x96 expected for this model), 
    and 3 represents the color channels (RGB).
    - audio (Tensor): A tensor of shape (audio_length, audio_level), where `audio_length` 
    is the length of the audio signal (audio_lenght == frames). Expected tensor shape is (audio_level, 1)

    Output:
    - output (Tensor): A tensor of shape (frames, 1), representing the binary classification score (0 or 1).
    """

    def __init__(self):
        super(Net_MN_Small_Lite, self).__init__()

        mobilenet = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        mobilenet.classifier = nn.Identity()
        self.video_cnn = mobilenet

        self.video_rnn = nn.GRU(input_size=576, hidden_size=512, num_layers=2, batch_first=True, bidirectional=True, dropout=0.3)

        self.audio_cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.AdaptiveAvgPool1d(1)
        )

        self.audio_rnn = nn.GRU(input_size=128, hidden_size=128, num_layers=2, batch_first=True, bidirectional=True, dropout=0.3)

        self.fc = nn.Sequential(
            nn.Linear(1024 + 256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, video, audio):
        video_features = self.video_cnn(video)
        video_features, _ = self.video_rnn(video_features)
        video_features = video_features.squeeze(0)

        audio = audio.permute(1, 0).unsqueeze(0)
        audio_features = self.audio_cnn(audio)
        audio_features = audio_features.squeeze(-1).unsqueeze(1)
        audio_features, _ = self.audio_rnn(audio_features)
        audio_features = audio_features.squeeze(0).repeat(video_features.shape[0], 1)
        
        combined_features = torch.cat((video_features, audio_features), dim=1)
        output = self.fc(combined_features)
        return output

class Net_MN_Small(nn.Module):

    def __init__(self):
        super(Net_MN_Small, self).__init__()

        mobilenet = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        mobilenet.classifier = nn.Identity()
        self.video_cnn = mobilenet

        self.video_rnn = nn.LSTM(input_size=576, hidden_size=512, num_layers=2, batch_first=True, bidirectional=True, dropout=0.3)

        self.audio_cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.AdaptiveAvgPool1d(1)
        )

        self.audio_rnn = nn.LSTM(input_size=256, hidden_size=256, num_layers=2, batch_first=True, bidirectional=True, dropout=0.3)


        self.fc = nn.Sequential(
            nn.Linear(1024 + 512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, video, audio):
        video_features = self.video_cnn(video)
        video_features, _ = self.video_rnn(video_features)
        video_features = video_features.squeeze(0)

        audio = audio.permute(1, 0).unsqueeze(0)
        audio_features = self.audio_cnn(audio)
        audio_features = audio_features.squeeze(-1).unsqueeze(1)
        audio_features, _ = self.audio_rnn(audio_features)
        audio_features = audio_features.squeeze(0).repeat(video_features.shape[0], 1)
        
        combined_features = torch.cat((video_features, audio_features), dim=1)
        output = self.fc(combined_features)
        return output

class Net_MN_Large(nn.Module):

    def __init__(self):
        super(Net_MN_Large, self).__init__()

        mobilenet = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
        mobilenet.classifier = nn.Identity()
        self.video_cnn = mobilenet

        self.video_rnn = nn.LSTM(input_size=960, hidden_size=512, num_layers=2, batch_first=True, bidirectional=True, dropout=0.3)

        self.audio_cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.AdaptiveAvgPool1d(1)
        )

        self.audio_rnn = nn.LSTM(input_size=256, hidden_size=256, num_layers=2, batch_first=True, bidirectional=True, dropout=0.3)

        self.fc = nn.Sequential(
            nn.Linear(1024 + 512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, video, audio):
        video_features = self.video_cnn(video)
        video_features, _ = self.video_rnn(video_features)
        video_features = video_features.squeeze(0)

        audio = audio.permute(1, 0).unsqueeze(0)
        audio_features = self.audio_cnn(audio)
        audio_features = audio_features.squeeze(-1).unsqueeze(1)
        audio_features, _ = self.audio_rnn(audio_features)
        audio_features = audio_features.squeeze(0).repeat(video_features.shape[0], 1)
        
        combined_features = torch.cat((video_features, audio_features), dim=1)
        output = self.fc(combined_features)
        return output

class Net_EN_B0(nn.Module):

    def __init__(self):
        super(Net_EN_B0, self).__init__()

        mobilenet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        mobilenet.classifier = nn.Identity()
        self.video_cnn = mobilenet

        self.video_rnn = nn.LSTM(input_size=1280, hidden_size=512, num_layers=2, batch_first=True, bidirectional=True, dropout=0.3)

        self.audio_cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.audio_rnn = nn.LSTM(input_size=256, hidden_size=256, num_layers=2, batch_first=True, bidirectional=True, dropout=0.3)

        self.fc = nn.Sequential(
            nn.Linear(1024 + 512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, video, audio):
        video_features = self.video_cnn(video)
        video_features, _ = self.video_rnn(video_features)
        video_features = video_features.squeeze(0)

        audio = audio.permute(1, 0).unsqueeze(0)
        audio_features = self.audio_cnn(audio)
        audio_features = audio_features.squeeze(-1).unsqueeze(1)
        audio_features, _ = self.audio_rnn(audio_features)
        audio_features = audio_features.squeeze(0).repeat(video_features.shape[0], 1)
        combined_features = torch.cat((video_features, audio_features), dim=1)
        output = self.fc(combined_features)
        return output