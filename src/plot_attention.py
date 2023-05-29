from .model_siamese_bert import (
    SiameseBertForSequenceClassification,
    MonoBertForSequenceClassification,
)
from transformers import (
    PretrainedConfig,
    utils
)
import torch
from torch.utils.data import (
    DataLoader,
)
from datasets import load_dataset, Array2D
from .tokenization_dna import DNATokenizer
import evaluate
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import argparse
import seaborn as sns


def save_heatmap_grid(attention, filename, coolwarm: bool = False):
    fig, axes = plt.subplots(12, 12, figsize=(60, 60))
    for i in range(12):
        for j in range(12):
            # print(f"\r{i}\t{j}", end='')
            ax = axes[i][j]
            kwargs = {}
            if i != 11:
                kwargs["xticklabels"] = []
            if j != 0:
                kwargs["yticklabels"] = []
            
            if coolwarm:
                sign = np.sign(attention[i][j, 1:-1, 1:-1])
                temp = sign * np.sqrt(sign * attention[i][j, 1:-1, 1:-1])
                vmin = np.min(temp)
                vmax = np.max(temp)
                divnorm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0., vmax=vmax)
                cmap_vlag = sns.color_palette("vlag", as_cmap=True)
                ax.imshow(temp, cmap=cmap_vlag, norm=divnorm)
            else:
                ax.imshow(np.sqrt(attention[i][j, 1:-1, 1:-1]), cmap="cubehelix_r")
            ax.set_title(f"Head:{j+1}-Layer:{i+1}, max:{np.max(attention[i][j, 1:-1, 1:-1]):.3f}")
            plt.setp(ax.spines.values(), alpha = 0)
            ax.tick_params(which = 'both', size = 0, labelsize = 0)
    plt.tight_layout()
    plt.savefig(filename)


def main():
    utils.logging.set_verbosity_error()

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True, help="path to the saved model (pytorch_model.bin)")
    parser.add_argument("--mono_bert", action="store_true", default=False, help="Set flag to use mono DNABERT base model")
    parser.add_argument("-o", "--output", required=True, help="output file name")

    args = parser.parse_args()

    torch.manual_seed(42)

    # pretrained = torch.load("/home/chuanyi/project/phylobert/data/models/patho2_freeze_lr_1e-05_seed_3/pytorch_model.bin")
    # pretrained = torch.load("/home/chuanyi/project/phylobert/data/models/mlm_snv_patho_large_freeze_lr_2e-05_seed-5_checkpoint-30000/pytorch_model.bin")
    # pretrained = torch.load("/home/chuanyi/project/phylobert/data/models/mlm_snv_patho_large_mono_freeze_lr_2e-05_seed-1_checkpoint-30000/pytorch_model.bin")
    pretrained = torch.load(args.model)
    bert_dict = {}
    for key in pretrained:
        if key.startswith("taskmodels_dict.clinvar2.bert"):
            bert_dict[key[len("taskmodels_dict.clinvar2.bert.") : ]] = pretrained[key]
    classifier_dict = {}
    for key in pretrained:
        if key.startswith("taskmodels_dict.clinvar2.classifier"):
            classifier_dict[key[len("taskmodels_dict.clinvar2.classifier.") : ]] = pretrained[key]
    config = PretrainedConfig.from_pretrained(
        "/home/chuanyi/project/phylobert/DNABERT/model/6-new-12w-0",
        num_labels=2,
    )
    config.output_attentions = True
    if args.mono_bert:
        siamese_bert = MonoBertForSequenceClassification(config=config)
    else:
        siamese_bert = SiameseBertForSequenceClassification(config=config)
    siamese_bert.bert.load_state_dict(bert_dict)
    siamese_bert.classifier.load_state_dict(classifier_dict)

    tokenizer = DNATokenizer.from_pretrained(
        "/home/chuanyi/project/phylobert/DNABERT/model/6-new-12w-0",
    )

    def convert_example_pairs_to_features(example_batch):
        inputs = list(example_batch['seq_mut'])
        features_mut = tokenizer.batch_encode_plus(
            inputs, max_length=512,
            add_special_tokens=True,
            truncation=True,
            padding='max_length'
        )

        inputs = list(example_batch['seq_ref'])
        features_ref = tokenizer.batch_encode_plus(
            inputs, max_length=512,
            add_special_tokens=True,
            truncation=True,
            padding='max_length',
        )
        
        features_mut["input_ids"] = [[mut, ref] for mut, ref in zip(features_mut["input_ids"], features_ref["input_ids"])]
        features_mut["attention_mask"] = [[mut, ref] for mut, ref in zip(features_mut["attention_mask"], features_ref["attention_mask"])]
        features_mut["labels"] = example_batch["label"]
        return features_mut

    text_dataset = load_dataset(
        "/home/chuanyi/project/phylobert/data/ClinVar",
        data_files={"eval": "dev.tsv"}
    )

    dataset = text_dataset["eval"].map(
        convert_example_pairs_to_features,
        batched=True,
        cache_file_name=f"/hdd/phylobert_data/cache/cache-clinvar-eval-{text_dataset['eval']._fingerprint}.arrow"
    )
    dataset.set_format(
        type="torch",
        columns=['input_ids', 'attention_mask', 'labels'],
    )
    dataset = dataset.cast_column("input_ids", feature=Array2D(shape=(2, -1), dtype="int32"))
    dataset = dataset.cast_column("attention_mask", feature=Array2D(shape=(2, -1), dtype="int8"))

    dataloader = DataLoader(
        dataset,
        batch_size=32,
        # sampler=RandomSampler,
        # collate_fn=DefaultDataCollator(),
    )

    metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    torch.manual_seed(42)
    siamese_bert.eval()
    # siamese_bert.bert.config.output_attentions = True
    with torch.no_grad():
        batch = next(dataloader.__iter__())
        if args.mono_bert:
            loss, logits, att0 = siamese_bert(**batch)
        else:
            loss, logits, att0, att1, _ = siamese_bert(**batch)
        print(np.argmax(logits.detach().numpy(), axis=1))
        print(batch["labels"].detach().numpy())

        base = args.output[:args.output.rfind('.')]

        attention0 = [np.exp(np.mean(np.log(a.detach().numpy()), axis=0)) for a in att0]
        save_heatmap_grid(attention0, base + '.ref_attention.png')

        if not args.mono_bert:
            attention1 = [np.exp(np.mean(np.log(a.detach().numpy()), axis=0)) for a in att1]
            save_heatmap_grid(attention1, base + '.mut_attention.png')

            diff = [a1 - a0 for a0, a1 in zip(attention0, attention1)]
            save_heatmap_grid(diff, base + '.diff_attention.png', coolwarm=True)

        # attention = [np.mean(a.detach().numpy(), axis=0) for a in att]
        # save_heatmap_grid(attention, "attentions_mean_batch32.png")


if __name__ == "__main__":
    main()
