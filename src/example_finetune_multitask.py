from transformers import (
    AutoConfig,
    AdamW,
    BertConfig,
    BertForSequenceClassification,
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
    MultitaskModel,
    MultitaskTrainer,
    FreezingCallback,
)
from .model_siamese_bert import (
    CustomBertForMaskedLM,
    MonoBertForSequenceClassification,
    SiameseBertForSequenceClassification,
)
from .tokenization_dna import DNATokenizer
from .utils import ModelArguments
from datasets import load_dataset, Dataset
import torch
from typing import Callable, Dict, List, Union, Any
import random
import numpy as np
from dataclasses import dataclass


# class CustomDataCollator(DefaultDataCollator):
#     """
#     Extending the existing DataCollator to work with NLP dataset batches
#     """
#     # # def collate_batch(self, features: List[Union[InputDataClass, Dict]]) -> Dict[str, torch.Tensor]:
#     # def collate_batch(self, features) -> Dict[str, torch.Tensor]:
#     #     first = features[0]
#     #     if isinstance(first, dict):
#     #       # NLP data sets current works presents features as lists of dictionary
#     #       # (one per example), so we  will adapt the collate_batch logic for that
#     #       if "labels" in first and first["labels"] is not None:
#     #           if first["labels"].dtype == torch.int64:
#     #               labels = torch.tensor([f["labels"] for f in features], dtype=torch.long)
#     #           else:
#     #               labels = torch.tensor([f["labels"] for f in features], dtype=torch.float)
#     #           batch = {"labels": labels}
#     #       for k, v in first.items():
#     #           if k != "labels" and v is not None and not isinstance(v, str):
#     #               batch[k] = torch.stack([f[k] for f in features])
#     #       return batch
#     #     else:
#     #       # otherwise, revert to using the default collate_batch
#     #       return DefaultDataCollator().collate_batch(features)

@dataclass
class MultitaskDataCollator():
    data_collator_dict: Dict[str, Any]
    return_tensors: str = 'pt'

    def __call__(self, features: List[Dict[str, Any]], return_tensors=None) -> Dict[str, Any]:
        if return_tensors is None:
            return_tensors = self.return_tensors
        return DefaultDataCollator().__call__(features, return_tensors)
        # TODO: make data collator compatible to all training sets
        # if len(features[0]["input_ids"].shape) == 2:
        #     return DefaultDataCollator().__call__(features, return_tensors)
        # else:
        #     return self.data_collator_dict["mlm"].__call__(features, return_tensors)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

def cast_func(features_dict, task_name, phase):
    features_dict[task_name][phase] = features_dict[task_name][phase].cast_column("input_ids", feature=Array2D(shape=(2, -1), dtype="int32"))
    features_dict[task_name][phase] = features_dict[task_name][phase].cast_column("attention_mask", feature=Array2D(shape=(2, -1), dtype="int8"))

@dataclass
class Task:
    model_class: BertPreTrainedModel
    config: Callable
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

    def convert_text_to_features(example_batch, label):
        inputs = list(example_batch[label])
        features = tokenizer.batch_encode_plus(
            inputs,
            max_length=args.max_seq_length,
            add_special_tokens=True,
            truncation=True,
            padding='max_length',
            return_special_tokens_mask=True,
        )
        return features

    if args.use_mono_bert:
        paired_bert_class = MonoBertForSequenceClassification
    else:
        paired_bert_class = SiameseBertForSequenceClassification
    model_dict: Dict[str, Task] = {
        "mlm": Task(
            model_class=CustomBertForMaskedLM,
            config=lambda path: PretrainedConfig.from_pretrained(
                path,
                num_labels=1,
            ),
            dataset=lambda: load_dataset(
                "/home/chuanyi/project/phylobert/DNABERT/examples/sample_data/pre",
                data_files={"train": "6_3k.txt"}
            ),
            convert_func=lambda x: convert_text_to_features(x, "text"),
            cast_func=lambda *_: None,
            columns=['input_ids', 'attention_mask', 'token_type_ids', 'special_tokens_mask'],
            data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15),
        ),
        "clinvar_snv": Task(
            model_class=paired_bert_class,
            config=lambda path: PretrainedConfig.from_pretrained(
                path,
                num_labels=2,
            ),
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
            model_class=paired_bert_class,
            config=lambda path: PretrainedConfig.from_pretrained(
                path,
                num_labels=2,
            ),
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
            model_class=paired_bert_class,
            config=lambda path: PretrainedConfig.from_pretrained(
                path,
                num_labels=2,
            ),
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
            model_class=paired_bert_class,
            config=lambda path: PretrainedConfig.from_pretrained(
                path,
                num_labels=2,
            ),
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
            model_class=paired_bert_class,
            config=lambda path: PretrainedConfig.from_pretrained(
                path,
                num_labels=2,
            ),
            dataset=lambda: load_dataset(
                "/home/chuanyi/project/phylobert/data/mc3/data_pheno",
                data_files={"train": "train.tsv", "eval": "dev.tsv"}
            ),
            convert_func=convert_example_pairs_to_features,
            cast_func=cast_func,
            columns=['input_ids', 'attention_mask', 'labels'],
            data_collator=DefaultDataCollator()
        ),
        "coding": Task(
            model_class=BertForSequenceClassification,
            config=lambda path: PretrainedConfig.from_pretrained(
                path,
                num_labels=2,
            ),
            dataset=lambda: load_dataset(
                "/home/chuanyi/project/phylobert/data/coding/data",
                data_files={"train": "train.tsv", "eval": "dev.tsv"},
            ),
            convert_func=lambda x: convert_text_to_features(x, "sequence"),
            cast_func=lambda *_: None,
            columns=['input_ids', 'attention_mask', 'label'],
            data_collator=DefaultDataCollator()
        ),
    }

    multitask_model = MultitaskModel.create(
        model_path=args.bert_model_path,
        model_type_dict={task: model_dict[task].model_class for task in args.tasks},
        model_config_dict={task: model_dict[task].config(args.bert_model_path) for task in args.tasks},
    )

    print(multitask_model.encoder.embeddings.word_embeddings.weight.data_ptr())
    for task_name, model in multitask_model.taskmodels_dict.items():
        print(model.bert.embeddings.word_embeddings.weight.data_ptr())

    dataset_dict: Dict[str, Dataset] = {task: model_dict[task].dataset() for task in args.tasks}

    metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])

    # Force `include_inputs_for_metrics` to be true for computing metrics
    train_args.include_inputs_for_metrics = True
    def compute_metrics(eval_pred: EvalPrediction):
        predictions, labels, inputs = eval_pred
        task_names = inputs[1]
        results = {}
        for task in np.unique(task_names):
            idx = task_names==task
            pred = np.argmax(predictions[idx], axis=1)
            result = metrics.compute(predictions=pred, references=labels[idx])
            result = {f"{task}_{k}": v for k, v in result.items()}
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

    # train_args = TrainingArguments(
    #     output_dir="./models/nvoeriuh",
    #     overwrite_output_dir=True,
    #     learning_rate=7e-5,
    #     do_train=True,
    #     do_eval=True,
    #     num_train_epochs=5.0,
    #     # Adjust batch size if this doesn't fit on the Colab GPU
    #     per_device_train_batch_size=8,
    #     per_device_eval_batch_size=16,
    #     evaluation_strategy='steps',
    #     eval_steps=200,
    #     save_steps=2000,
    #     warmup_ratio=0.1,
    #     weight_decay=0.01,
    #     include_inputs_for_metrics=True,
    #     label_names=["labels"],
    # )

    trainer = MultitaskTrainer(
        model=multitask_model,
        args=train_args,
        data_collator=MultitaskDataCollator(data_collator_dict),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )
    trainer.add_callback(FreezingCallback(trainer, args.freeze_ratio))

    trainer.train()

    # preds_dict = {}
    # for task_name in args.tasks:
    #     eval_dataloader = DataLoaderWithTaskname(
    #         task_name,
    #         trainer.get_eval_dataloader(eval_dataset=features_dict[task_name]["validation"])
    #     )
    #     print(eval_dataloader.data_loader.collate_fn)
    #     preds_dict[task_name] = trainer._prediction_loop(
    #         eval_dataloader, 
    #         description=f"Validation: {task_name}",
    #     )

if __name__ == "__main__":
    main()
