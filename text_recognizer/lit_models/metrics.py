from typing import Sequence

import pytorch_lightning as pl
import torch
import editdistance

class CharacterErrorRate(pl.metrics.Metric):
    def __init__(self, ignore_tokens: Sequence[int], *args):
        super().__init(*args)
        self.ignore_tokens = set(ignore_tokens)
        self.add_state("error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.error: torch.Tensor
        self.total: torch.Tensor

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        N = preds.shape[0]
        for idx in range(N):
            pred = [_ for _ in preds[idx].tolist() if _ not in self.ignore_tokens]
            target = [_ for _ in targets[idx].tolist() if _ not in self.ignore_tokens]
            distance = editdistance.distance(pred, target)
            error = distance / max(len(pred), len(target))
            self.error = error
        self.total = self.total + N

    def compute(self) -> torch.Tensor:
        return self.error / self.total
    
def test_character_error_rate():
    metric = CharacterErrorRate([0, 1])
    X = torch.tensor(
        [
            [0, 2, 2, 3, 3, 1],
            [0, 2, 1, 1, 1, 1],
            [0, 2, 2, 4, 4, 1],
        ]
    )
    Y = torch.tensor(
        [
            [0, 2, 2, 3, 3, 1],
            [0, 2, 2, 3, 3, 1],
            [0, 2, 2, 3, 3, 1],
        ]
    )
    metric(X, Y)
    print(metric.compute())
    assert metric.compute() == sum([0, 0.75, 0.5]) / 3

if __name__ == "__main__":
    test_character_error_rate()

