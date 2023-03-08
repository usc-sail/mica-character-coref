from torch import nn
import torch

class PairwiseEncoder(nn.Module):
    def __init__(self, dropout: float):
        super().__init__()
        self.genre2int = {
            g: gi for gi, g in enumerate(
                ["bc", "bn", "mz", "nw", "pt", "tc", "wb"])}
        self.genre_emb = nn.Embedding(len(self.genre2int), 20)

        # each position corresponds to a bucket:
        #   [(0, 2), (2, 3), (3, 4), (4, 5), (5, 8),
        #    (8, 16), (16, 32), (32, 64), (64, float("inf"))]
        self.distance_emb = torch.nn.Embedding(9, 20)

        # two possibilities: same vs different speaker
        self.speaker_emb = torch.nn.Embedding(2, 20)

        self.dropout = nn.Dropout(dropout)
        self._device = torch.device("cpu")

    @property
    def device(self) -> torch.device:
        return self._device
    
    @device.setter
    def device(self, device: torch.device):
        self.to(device)
        self._device = device

    def forward(self,  
                top_indices: torch.Tensor,
                speakers: list[str],
                genre: str,
                word_ids: torch.Tensor = None) -> torch.Tensor:
        if word_ids is None:
            word_ids = torch.arange(0, len(top_indices), device=self.device)
        speaker_map = torch.tensor(self._speaker_map(speakers), device=self.device)
        same_speaker = (speaker_map[top_indices] == speaker_map[word_ids].unsqueeze(1))
        same_speaker = self.speaker_emb(same_speaker.to(torch.long))

        # bucketing the distance (see __init__())
        distance = (word_ids.unsqueeze(1) - top_indices).abs_().clamp_min_(min=1)
        log_distance = distance.to(torch.float).log2().floor_()
        log_distance = log_distance.clamp_max_(max=6).to(torch.long)
        distance = torch.where(distance < 5, distance - 1, log_distance + 2)
        distance = self.distance_emb(distance)

        genre = torch.tensor(self.genre2int[genre], device=self.device).expand_as(top_indices)
        genre = self.genre_emb(genre)

        return self.dropout(torch.cat((same_speaker, distance, genre), dim=2))

    def _speaker_map(self, speakers: list[str]) -> list[int]:
        str2int = {s: i for i, s in enumerate(set(speakers))}
        return [str2int[s] for s in speakers]
