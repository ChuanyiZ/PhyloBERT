from transformers import (
    AutoConfig,
    AdamW,
    BertConfig,
    BertPreTrainedModel,
    DataCollator,
    DefaultDataCollator,
    DataCollatorForLanguageModeling,
    PretrainedConfig,
    TrainingArguments,
    EvalPrediction,
    HfArgumentParser,
)
import evaluate
from datasets import (
    Features,
    Value,
    Array2D,
    Sequence,
)
# from transformers.data.data_collator import DataCollator, InputDataClass, DefaultDataCollator
from .model_multitask import (
    MultitaskTrainer,
    MultitaskTrainerFixed,
    FreezingCallback,
)
from .model_siamese_bert import (
    CustomBertForMaskedLM,
    # SiameseBertForSequenceClassification,
    MutBertForSequenceClassification,
)
from .tokenization_dna import DNATokenizer
from .utils import ModelArguments
from datasets import load_dataset, Dataset
import torch
from typing import Callable, Dict, List, Union, Any
import random
import numpy as np
from dataclasses import dataclass


@dataclass
class MultitaskDataCollator():
    data_collator_dict: Dict[str, Any]
    return_tensors: str = 'pt'

    def __call__(self, features: List[Dict[str, Any]], return_tensors=None) -> Dict[str, Any]:
        if return_tensors is None:
            return_tensors = self.return_tensors
        if len(features[0]["input_ids"].shape) == 2:
            return DefaultDataCollator().__call__(features, return_tensors)
        else:
            return self.data_collator_dict["mlm"].__call__(features, return_tensors)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def cast_func(features_dict, task_name, phase):
    features_dict[task_name][phase] = features_dict[task_name][phase].cast_column("input_ids", feature=Array2D(shape=(2, -1), dtype="int32"))
    features_dict[task_name][phase] = features_dict[task_name][phase].cast_column("attention_mask", feature=Array2D(shape=(2, -1), dtype="int8"))


@dataclass
class Task:
    dataset: Callable
    convert_func: Callable
    cast_func: Callable
    columns: List
    data_collator: DataCollator


def main():
    # args = get_args()
    parser = HfArgumentParser((TrainingArguments, ModelArguments))
    train_args, args = parser.parse_args_into_dataclasses()
    train_args.optim="adamw_torch"
    train_args.report_to=["tensorboard"]

    set_seed(train_args)

    tokenizer = DNATokenizer.from_pretrained(
        args.bert_model_path,
        do_lower_case=args.do_lower_case,
    )

    def convert_example_pairs_to_features(example_batch):
        inputs = list(example_batch['seq_mut'])
        features_mut = tokenizer.batch_encode_plus(
            inputs, max_length=args.max_seq_length,
            add_special_tokens=True,
            truncation=True,
            padding='max_length'
        )

        inputs = list(example_batch['seq_ref'])
        features_ref = tokenizer.batch_encode_plus(
            inputs, max_length=args.max_seq_length,
            add_special_tokens=True,
            truncation=True,
            padding='max_length',
        )
        
        features_mut["input_ids"] = [[mut, ref] for mut, ref in zip(features_mut["input_ids"], features_ref["input_ids"])]
        features_mut["attention_mask"] = [[mut, ref] for mut, ref in zip(features_mut["attention_mask"], features_ref["attention_mask"])]
        features_mut["labels"] = example_batch["label"]
        return features_mut

    model_dict: Dict[str, Task] = {
        "clinvar_snv": Task(
            dataset=lambda: load_dataset(
                "/home/chuanyi/project/phylobert/data/ClinVar/data_snv",
                data_files={"train": "train.tsv", "eval": "dev.tsv"},
            ),
            convert_func=convert_example_pairs_to_features,
            cast_func=cast_func,
            columns=['input_ids', 'attention_mask', 'labels'],
            data_collator=DefaultDataCollator()
        ),
        "clinvar": Task(
            dataset=lambda: load_dataset(
                "/home/chuanyi/project/phylobert/data/ClinVar",
                data_files={"train": "train.tsv", "eval": "dev.tsv"}
            ),
            convert_func=convert_example_pairs_to_features,
            cast_func=cast_func,
            columns=['input_ids', 'attention_mask', 'labels'],
            data_collator=DefaultDataCollator()
        ),
        "clinvar2": Task(
            dataset=lambda: load_dataset(
                "/home/chuanyi/project/phylobert/data/ClinVar2/data_patho",
                data_files={"train": "train.tsv", "eval": "dev.tsv"}
            ),
            convert_func=convert_example_pairs_to_features,
            cast_func=cast_func,
            columns=['input_ids', 'attention_mask', 'labels'],
            data_collator=DefaultDataCollator()
        ),
        "mc3_consequence": Task(
            dataset=lambda: load_dataset(
                "/home/chuanyi/project/phylobert/data/mc3/data_consequence",
                data_files={"train": "train.tsv", "eval": "dev.tsv"}
            ),
            convert_func=convert_example_pairs_to_features,
            cast_func=cast_func,
            columns=['input_ids', 'attention_mask', 'labels'],
            data_collator=DefaultDataCollator()
        ),
        "mc3_pheno": Task(
            dataset=lambda: load_dataset(
                "/home/chuanyi/project/phylobert/data/mc3/data_pheno",
                data_files={"train": "train.tsv", "eval": "dev.tsv"}
            ),
            convert_func=convert_example_pairs_to_features,
            cast_func=cast_func,
            columns=['input_ids', 'attention_mask', 'labels'],
            data_collator=DefaultDataCollator()
        ),
    }

    multitask_model = MutBertForSequenceClassification(
        PretrainedConfig.from_pretrained(
            args.bert_model_path,
            num_labels=2,
        ),
        num_task=len(args.tasks),
    )

    dataset_dict: Dict[str, Dataset] = {task: model_dict[task].dataset() for task in args.tasks}

    metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])

    # Force `include_inputs_for_metrics` to be true for computing metrics
    train_args.include_inputs_for_metrics = True
    def compute_metrics(eval_pred: EvalPrediction):
        predictions, labels, inputs = eval_pred
        task_ids = inputs[1]
        results = {}
        for task in np.unique(task_ids):
            idx = task_ids==task
            pred = np.argmax(predictions[idx], axis=1)
            result = metrics.compute(predictions=pred, references=labels[idx])
            task_name = args.tasks[task]
            result = {f"{task_name}_{k}": v for k, v in result.items()}
            results.update(result)
        return results

    features_dict = {}
    for task_name, dataset in dataset_dict.items():
        features_dict[task_name] = {}
        for phase, phase_dataset in dataset.items():
            features_dict[task_name][phase] = phase_dataset.map(
                model_dict[task_name].convert_func,
                batched=True,
                # features=features_dict[task_name],
                cache_file_name=f"/home/chuanyi/project/phylobert/cache/cache-{task_name}-{phase}-{phase_dataset._fingerprint}.arrow"
            )
            print(task_name, phase, len(phase_dataset), len(features_dict[task_name][phase]))
            model_dict[task_name].cast_func(features_dict, task_name, phase)
            features_dict[task_name][phase].set_format(
                type="torch",
                columns=model_dict[task_name].columns,
            )
            print(task_name, phase, len(phase_dataset), len(features_dict[task_name][phase]))

    train_dataset = {
        task_name: dataset["train"]
        for task_name, dataset in features_dict.items()
    }

    eval_dataset = {
        task_name: dataset["eval"]
        for task_name, dataset in features_dict.items()
        if "eval" in dataset
    }

    data_collator_dict = {task: model_dict[task].data_collator for task in args.tasks}

    trainer = MultitaskTrainerFixed(
        model=multitask_model,
        args=train_args,
        data_collator=MultitaskDataCollator(data_collator_dict),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        task_names=args.tasks,
    )
    trainer.add_callback(FreezingCallback(trainer, args.freeze_ratio))

    trainer.train()


if __name__ == "__main__":
    main()
