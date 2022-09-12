from dataclasses import dataclass, field
from typing import List

@dataclass
class ModelArguments:
    bert_model_path: str = field(
        metadata={"help": "The directory storing the base BERT encoder."},
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
