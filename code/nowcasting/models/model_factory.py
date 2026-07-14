import torch
from nowcasting.models.registry import build_model

class Model(object):
    def __init__(self, configs):
        self.configs = configs
        self.data_frame = []
        self.network = build_model(configs).to(configs.device)
        self.test_load()

    def test_load(self):
        stats = torch.load(self.configs.pretrained_model, map_location=self.configs.device)
        if isinstance(stats, dict) and 'model' in stats:
            stats = stats['model']
        self.network.load_state_dict(stats)

    def test(self, frames):
        frames_tensor = torch.FloatTensor(frames).to(self.configs.device)
        self.network.eval()
        with torch.no_grad():
            next_frames = self.network(frames_tensor)
        return next_frames.detach().cpu().numpy()
