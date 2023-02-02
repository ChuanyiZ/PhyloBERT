from dataclasses import dataclass, field
from typing import List
import bisect
from torch.utils.data import ConcatDataset

@dataclass
class ModelArguments:
    bert_model_path: str = field(
        metadata={"help": "The directory storing the base BERT encoder."},
    )
    bert_model_config_path: str = field(
        default=None,
        metadata={"help": "The path to the configuration JSON file."},
    )
    do_lower_case: bool = field(
        default=True,
        metadata={"help": "Set this flag if you are using an uncased model."},
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. "
                "Sequences longer than this will be truncated, sequences shorter "
                "will be padded."
            )
        }
    )
    tasks: List[str] = field(
        default_factory=lambda: ["mlm", "clinvar_snv", "clinvar"],
        metadata={
            "nargs": '+',
            "help": (
                "Training tasks"
            )
        }
    )

    freeze_steps: int = field(
        default=0,
        metadata={"help": "Freeze part of the model in freeze_steps."},
    )
    freeze_ratio: float = field(
        default=0,
        metadata={"help": "Freeze part of the model in freeze_ratio*total_steps."},
    )
    freeze_layers: str = field(
        default=None,
        metadata={"help": "A list of prefixes of layers to be frozen."},
    )

    use_mono_bert: bool = field(
        default=False,
        metadata={"help": "Set this flag if using mono BERT."},
    )

class MultitaskDataset(ConcatDataset):
    # def __init__(self, datasets):
    #     super(MyConcatDataset, self).__init__(datasets)

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        data = self.datasets[dataset_idx][sample_idx]
        data["task_ids"] = dataset_idx
        return data
