from typing import Any, Callable, Dict, Sequence, Tuple, Union
import torch

class BaseDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data: Union[Sequence, torch.Tensor],
                 targets: Union[Sequence, torch.Tensor],
                 transform: Callable = None,
                 target_transform: Callable = None) -> None:
        if len(data) != len(targets):
            raise ValueError("Data and Target length mismatch")
        super.__init__()
        self.data = data
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        datum, target = self.data[index], self.targets[index]

        if self.transform is not None:
            datum = self.transform(datum)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return datum, target
    
def convert_strings_to_labels(strings: Sequence[str],
                                  mapping: Dict[str, int],
                                  length: int) -> torch.Tensor:
    labels = torch.ones((len(strings), length), dtype=torch.long) * mapping["<P>"]

    for i, string in enumerate(strings):
        tokens = list(string)
        tokens = ["<S>", *tokens, "<E>"]
        for ii, token in enumerate(tokens):
            labels[i, ii] = mapping[token]
    return labels
    
def split_dataset(base_dataset: BaseDataset,
                  fraction: float,
                  seed: int) -> Tuple[BaseDataset, BaseDataset]:
    size_a = int(fraction*len(base_dataset))
    size_b = len(base_dataset) - size_a

    return torch.utils.data.random_split(
        base_dataset, 
        [size_a, size_b],
        generator=torch.Generator().manual_seed(seed)
        )