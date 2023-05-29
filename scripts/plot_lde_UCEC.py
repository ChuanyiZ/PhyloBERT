import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import umap
from sklearn.preprocessing import StandardScaler
import argparse
from tqdm import tqdm
import os

def main(args):
    sns.set_context("notebook", font_scale=1.5)

    TCGA_projects = [
        "UCEC",
    ]

    # rely on lazy binding of "df_vep"
    features_template = {
        "STRAND": {
            "hue": lambda: df_vep["STRAND"][idx],
            "palette": "vlag"
        },
        "SIFT_score": {
            "hue": lambda: df_vep["SIFT_score"][idx],
            "palette": "vlag"
        },
        "PolyPhen_score": {
            "hue": lambda: df_vep["PolyPhen_score"][idx],
            "palette": "vlag"
        },
        "Consequence": {
            "hue": lambda: np.where(df_vep["Consequence"][idx].isin({"missense_variant", "synonymous_variant", "stop_gained"}), df_vep["Consequence"][idx], np.nan),
        },
        "IMPACT": {
            "hue": lambda: df_vep["IMPACT"][idx]
        },
        "INTRON": {
            "hue": lambda: df_vep["INTRON"][idx].isna()
        },
        "Variant": {
            "hue": lambda: df_vep.index[idx].str.slice(start=-3)
        },
        "Allele":  {
            "hue": lambda: df_vep["Allele"][idx]
        },
        # "subtype": {
        #     "hue": lambda: df_vep["subtype"][idx],
        # },
    }

    df_ddr = pd.read_excel('/home/chuanyi/TCGA_DDR_Data_Resources.xlsx', sheet_name='DDR footprints', skiprows=3)

    subtypes = df_ddr.loc[df_ddr["disease"] == "UCEC", "subtype"].unique()

    fig, axes = plt.subplots(
        nrows=5,
        ncols=len(features_template),
        figsize=(9*len(features_template), 9*5)
    )
    for idx_, project in enumerate(tqdm(TCGA_projects)):
        
        df_mut = pd.read_csv(
            f"{args.path}/TCGA_{project}.tsv.gz",
            sep='\t',
            usecols=["index", "Sample_ID", "gene", "chrom", "start", "end", "ref", "alt"]
        )
        df_mut["Uploaded_variation"] = df_mut["chrom"] + '_' + df_mut["start"].astype(str) + '_' + df_mut["ref"] + '/' + df_mut["alt"]

        # remove duplicated
        embeddings = np.load(f"{args.path}/TCGA_{project}.embedding_{args.pooling_method}.npy")
        embeddings = embeddings[~df_mut["Uploaded_variation"].duplicated(), :]
        df_mut = df_mut.loc[~df_mut["Uploaded_variation"].duplicated(), :]
        if not args.is_mutbert:
            embeddings = np.hstack([embeddings, np.abs(embeddings[:, :768] - embeddings[:, 768:])])

        df_vep = pd.read_csv(f"/hdd/phylobert_data/embeddings/TCGA_{project}.vep.tsv",
                            sep='\t', comment="#",
                            na_values=['-'],
                            # dtype={"STRAND": float},
                            names=["Uploaded_variation", "Location", "Allele", "Gene", "Feature",
                                    "Feature_type", "Consequence", "cDNA_position", "CDS_position",
                                    "Protein_position", "Amino_acids", "Codons", "Existing_variation",
                                    "IMPACT", "DISTANCE", "STRAND", "FLAGS", "VARIANT_CLASS", "SYMBOL",
                                    "SYMBOL_SOURCE", "HGNC_ID", "BIOTYPE", "CANONICAL", "MANE_SELECT",
                                    "MANE_PLUS_CLINICAL", "TSL", "APPRIS", "CCDS", "ENSP", "SWISSPROT",
                                    "TREMBL", "UNIPARC", "UNIPROT_ISOFORM", "GENE_PHENO", "SIFT",
                                    "PolyPhen", "EXON", "INTRON", "DOMAINS", "miRNA", "HGVSc", "HGVSp",
                                    "HGVS_OFFSET", "AF", "AFR_AF", "AMR_AF", "EAS_AF", "EUR_AF", "SAS_AF",
                                    "AA_AF", "EA_AF", "gnomAD_AF", "gnomAD_AFR_AF", "gnomAD_AMR_AF",
                                    "gnomAD_ASJ_AF", "gnomAD_EAS_AF", "gnomAD_FIN_AF", "gnomAD_NFE_AF",
                                    "gnomAD_OTH_AF", "gnomAD_SAS_AF", "MAX_AF", "MAX_AF_POPS", "CLIN_SIG",
                                    "SOMATIC", "PHENO", "PUBMED", "MOTIF_NAME", "MOTIF_POS", "HIGH_INF_POS",
                                    "MOTIF_SCORE_CHANGE", "TRANSCRIPTION_FACTORS"])
        df_vep = df_vep.loc[~df_vep["Uploaded_variation"].duplicated(), :]
        df_vep = df_vep.set_index("Uploaded_variation").loc[df_mut["Uploaded_variation"], :]
        df_vep["miRNA"] = df_vep["miRNA"].fillna('')
        pattern = re.compile(r'.*\(([\d\.]+)\)')
        df_vep["SIFT_score"] = df_vep["SIFT"].apply(lambda x: re.search(pattern, x).group(1) if isinstance(x, str) else float("NaN")).astype(float)
        df_vep["PolyPhen_score"] = df_vep["PolyPhen"].apply(lambda x: re.search(pattern, x).group(1) if isinstance(x, str) else float("NaN")) .astype(float)

        df_vep["subtype"] = df_ddr.set_index("TCGA sample barcode").reindex(df_mut["Sample_ID"].str.slice(start=0, stop=-1))["subtype"].values

        for idx_subtype, subtype in enumerate(tqdm(subtypes, leave=True)):
            idx = (df_vep["subtype"] == subtype).values.nonzero()[0]
            if len(idx) > 200000:
                idx = idx[::(len(idx) // 200000)]
            
            model = args.path.strip('/').split('/')[-1]
            lde_filename = f".UMAP_LDE_subtype.project_{project}.subtype_{subtype}.model_{model}_{args.pooling_method}.npy"
            if os.path.isfile(lde_filename):
                lde = np.load(lde_filename)
            else:
                lde = umap.UMAP(random_state=42, low_memory=True).fit_transform(StandardScaler().fit_transform(embeddings[idx, :]))
                np.save(lde_filename, lde, allow_pickle=False)


            features = {
                feature: {
                    k: (v() if callable(v) and v.__name__ == "<lambda>" else v)
                    for k, v in d.items()
                }
                for feature, d in features_template.items()
            }

            for idx_feature, (feature, kwargs) in enumerate(tqdm(features.items(), leave=False)):
                if isinstance(axes, np.ndarray):
                    if len(axes.shape) == 2:
                        ax = axes[idx_subtype, idx_feature]
                    elif len(axes.shape) == 1:
                        if len(TCGA_projects) == 1:
                            ax = axes[idx_feature]
                        else:
                            ax = axes[idx_subtype]
                else:
                    ax = axes

                if lde.shape[0] < 80000:
                    alpha = 0.4
                else:
                    alpha = 0.1
                sns.scatterplot(
                    x=lde[:, 0],
                    y=lde[:, 1],
                    alpha=alpha,
                    s=1,
                    ax=ax,
                    **kwargs
                )
                ax.set_title(feature)
                if idx_feature == 0:
                    ax.set_ylabel(subtype)
                else:
                    ax.set_yticklabels([])
            plt.tight_layout()
        plt.savefig(args.output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot low dimmension embedding of a DNABERT model's embeddings")
    parser.add_argument("-o", "--output", help="output figure name")
    parser.add_argument("-p", "--path", help="path to the folder containing DNABERT embeddings")
    parser.add_argument("--is-mutbert", action="store_true", default=False, help="Set flag to use MutBERT")
    parser.add_argument("--pooling-method", default="CLS", choices=["CLS", "average"],
                        help="Pooling method")
    args = parser.parse_args()
    main(args)
