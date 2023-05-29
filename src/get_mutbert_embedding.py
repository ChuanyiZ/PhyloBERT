#!/usr/bin/env python
# coding: utf-8


import os
import pandas as pd
import numpy as np
import sys
import torch
import numpy as np
import csv
import argparse
import time
import datetime
import logging
import random
from pysam import FastaFile
from torch.utils.data import DataLoader
from .tokenization_dna import DNATokenizer
from transformers import PretrainedConfig
from .model_siamese_bert import SiameseBertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from typing import List, Dict
import functools
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Get DNABERT embedding")
parser.add_argument("-i", "--input", help="Input TSV file")
parser.add_argument("--input-vep", help="Input VEP file")
parser.add_argument("-m", "--model",
                    help="Path to the pretrained model (.bin)",
                    default="/home/chuanyi/project/phylobert/data/models/pretrain_patho_snv_patho_mc3_conseq_freeze_lr_2e-05_seed-6_checkpoint-75000/pytorch_model.bin")
parser.add_argument("-o", "--output", help="Output prefix")
args = parser.parse_args()

base_dir = "/home/rohan/rohan/mc3/"
REFERENCE = base_dir + "GRCh38_full_analysis_set_plus_decoy_hla.fa"
seq_length=510
logger = logging.getLogger('mc3_to_seq')

seed = 0
random.seed(seed)
np.random.seed(seed)


class get_seq(object):
    def __init__(self, seq, ref, alt, index):
        self.seq =  seq
        self.size = len(self.seq)
        self.index = int(index) -1 
        self.ref =  ref
        self.alt = alt
        self.ref_size = len(self.ref)
        
    def __iter__(self):
        return self

    def get_ref(self):
        ref_index = self.index + self.ref_size
        str_ref = self.seq[self.index:ref_index]
        if str(str_ref) == str(self.ref):
            return True
        else:
            print("Warning: Ref seq base {0} does not match the ref in vcf {1} for seq {2} for index {3}".format(str_ref, 
                                                                                                                 self.ref,
                                                                                                                 self.seq, self.index))

    def generate_mutant(self):
        check_ref = self.get_ref()
        mut_seq = str(self.seq[:self.index]) + str(self.alt) + str(self.seq[self.index + 1:])
        return mut_seq

    def generate_del(self):
        mut_seq =  str(self.seq[:self.index + 1]) + str(self.seq[self.index + self.ref_size :])
        return mut_seq


def reference_sequence_del(reference, 
                           seq_length, 
                           chrom, 
                           pos):
    genome = FastaFile(reference)
    pad = int(seq_length/2)
    #if deletion_length:
    start =  int(pos) - pad
    ex_stop = int(pos) + pad + int(deletion_length)
    stop = int(pos) + pad
    del_seq = genome.fetch(chrom, start, ex_stop)
    seq = genome.fetch(chrom, start, stop)
    return(del_seq, seq)

def reference_sequence(reference, seq_length, chrom, pos):
    genome = FastaFile(reference)
    pad = int(seq_length/2)
    start =  int(pos) - pad
    stop = int(pos) + pad
    seq = genome.fetch(chrom, start, stop)
    return seq

def reference_sequence_random(reference, 
                              seq_length, 
                              chrom, 
                              pos, logger,
                              deletion_length=False):
    genome = FastaFile(reference)
    insert = random.randint(1, seq_length)
    logger.info("position of the insert is {0}".format(insert))
    pad_5 = int(seq_length) 
    pad_3 = int(seq_length) - insert
    if deletion_length:
        start =  int(pos) - insert
        ex_stop = int(pos) + pad_3 + int(deletion_length)
        stop = int(pos) + pad_3
        del_seq = genome.fetch(chrom, start, ex_stop)
        seq = genome.fetch(chrom, start, stop)
        return(del_seq, seq, insert)
    else:
        start =  int(pos) - insert
        stop = int(pos) + pad_3
        seq = genome.fetch(chrom, start, stop)
        return(seq, insert)

def sliding_windown(seq, kmer):
    seq_str = ''
    for i in range(0,len(seq),1):
        kmer_str = str(seq[i:i + kmer])
        if len(kmer_str) != kmer:
            pass
        else:
            seq_str += (kmer_str) + ' '
    return seq_str


def execute_kmers_del(reference, seq_length,chrom, pos, ref, alt, logger):
        ext_seq, ref_seq, index = reference_sequence_del(reference, seq_length, chrom, pos)
        get_results = get_seq(ext_seq, ref, alt, index)
        mutant_seq = get_results.generate_del()
        logger.info("{0}, {1}, {2}, {3}, {4}, {5}".format(chrom, 
                                                          pos, 
                                                          ref, 
                                                          alt, 
                                                          ref_seq, mutant_seq))
        return ref_seq, mutant_seq

COMPLEMENT = {
    a: b for a, b in zip(
        "ACTGMYKRSWVHDBNactgmykrswvhdbn",
        "TGACKRMYWSBDHVNtgackrmywsbdhvn"
    )
}

def reverse_complement(seq: str) -> str:
    return ''.join(COMPLEMENT[x] for x in seq[::-1])

def execute_kmers(reference, seq_length, chrom, pos, ref, alt, logger, strand=0):
        # ref_seq, index = reference_sequence_random(reference, seq_length,
        #                                     chrom, pos, logger)
        ref_seq = reference_sequence(reference, seq_length, chrom, pos)
        index = int(seq_length/2)
        #print(ref_seq)
        get_results = get_seq(ref_seq, ref, alt, index)
        mutant_seq = get_results.generate_mutant()
        #logger.info("{0}, {1}, {2}, {3}, {4}, {5}".format(chrom, 
        #                                                  pos, 
        #                                                  ref, 
        #                                                  alt, 
        #
        #print(ref_seq, mutant_seq, index)
        if strand < 0:
            ref_seq = reverse_complement(ref_seq)
            mutant_seq = reverse_complement(mutant_seq)
        return ref_seq, mutant_seq, index

def get_ref(reference, seq_length, chrom, pos, ref, alt, logger, strand=0):
    ref_seq, mutant_seq, index = execute_kmers(reference, seq_length, chrom, pos, ref, alt, logger, strand)
    #print(ref_seq)
    return str(ref_seq)
    
def get_mut(reference, seq_length, chrom, pos, ref, alt, logger, strand=0):
    ref_seq, mutant_seq, index = execute_kmers(reference, seq_length, chrom, pos, ref, alt, logger, strand)
    return str(mutant_seq)

def get_index(reference, seq_length, chrom, pos, ref, alt, logger, strand=0):
    ref_seq, mutant_seq, index = execute_kmers(reference, seq_length, chrom, pos, ref, alt, logger, strand)
    return int(index)
    
def split_dataframe(df, chunk_size = 100):
    chunks = list()
    num_chunks = len(df) // chunk_size + 1
    for i in range(num_chunks):
        chunks.append(df[i*chunk_size:(i+1)*chunk_size])
    return chunks

class SequenceDataset(Dataset):  # type: ignore[type-arg]
    """Dataset initialized from a list of sequence strings."""

    def __init__(
        self,
        sequences: List[List[str]],
        seq_length: int,
        tokenizer,
        kmer_size: int = 6,
        verbose: bool = True,
    ):
        self.batch_encodings = self.tokenize_sequences(
            sequences, tokenizer, seq_length, kmer_size, verbose
        )

    @staticmethod
    def tokenize_sequences(
        sequences: List[List[str]],
        tokenizer,
        seq_length: int,
        kmer_size: int = 6,
        verbose: bool = True,
    ) -> List:

        tokenizer_fn = functools.partial(
            tokenizer.encode_plus,
            max_length=seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        batch_encodings = [
            [
                tokenizer_fn(SequenceDataset.group_by_kmer(seq1, kmer_size)),
                tokenizer_fn(SequenceDataset.group_by_kmer(seq2, kmer_size)),
            ]
            for seq1, seq2 in tqdm(sequences, desc="Tokenizing...", disable=not verbose)
        ]
        return batch_encodings

    @staticmethod
    def group_by_kmer(seq: str, kmer: int, stride: int = 1) -> str:
        return " ".join(seq[i : i + kmer] for i in range(0, len(seq), stride)).upper()

    def __len__(self) -> int:
        return len(self.batch_encodings)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        batch_encoding = self.batch_encodings[idx]
        # Squeeze so that batched tensors end up with (batch_size, seq_length)
        # instead of (batch_size, 1, seq_length)
        sample = {
            "input_ids": torch.stack(
                [batch_encoding[i]["input_ids"].squeeze() for i in range(len(batch_encoding))]),
            "attention_mask": torch.stack(
                [batch_encoding[i]["attention_mask"].squeeze() for i in range(len(batch_encoding))]),
            # "input_ids": batch_encoding["input_ids"].squeeze(),
            # "attention_mask": batch_encoding["attention_mask"],
        }
        return sample


pretrained = torch.load(args.model)
# pretrained = torch.load("/home/chuanyi/project/phylobert/data/models/snv_patho_mc3_conseq_freeze_lr_2e-05_seed-6_checkpoint-75000/pytorch_model.bin")
# pretrained = torch.load("/home/chuanyi/project/phylobert/data/models/mlm_snv_patho_large_mono_freeze_lr_2e-05_seed-1_checkpoint-30000/pytorch_model.bin")
bert_dict = {}
for key in pretrained:
    if key.startswith("taskmodels_dict.clinvar2_snv.bert"):
        bert_dict[key[len("taskmodels_dict.clinvar2_snv.bert.") : ]] = pretrained[key]
classifier_dict = {}
for key in pretrained:
    if key.startswith("taskmodels_dict.clinvar2_snv.classifier"):
        classifier_dict[key[len("taskmodels_dict.clinvar2_snv.classifier.") : ]] = pretrained[key]

config = PretrainedConfig.from_pretrained(
    "/home/chuanyi/project/phylobert/DNABERT/model/6-new-12w-0",
    num_labels=2,
)
config.output_attentions = True
siamese_bert = SiameseBertForSequenceClassification(config=config)
siamese_bert.bert.load_state_dict(bert_dict)
siamese_bert.classifier.load_state_dict(classifier_dict)
siamese_bert.to("cuda")

tokenizer = DNATokenizer.from_pretrained(
    "/home/chuanyi/project/phylobert/DNABERT/model/6-new-12w-0",
)

TCGA_df = pd.read_csv(args.input, sep='\t')
TCGA_nodels_df = TCGA_df[(TCGA_df['ref'] != "-") & (TCGA_df['alt'] != "-") ] 
TCGA_nodels_df = TCGA_nodels_df.reset_index().copy()
TCGA_nodels_df['ref_seq'] = None
TCGA_nodels_df['mut_seq'] = None
TCGA_nodels_df['mutant_index'] = None

if args.input_vep:
    df_vep = pd.read_csv(args.input_vep,
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
    TCGA_nodels_df["strand"] = np.concatenate([df_vep["STRAND"].values, np.ones(len(TCGA_nodels_df) - len(df_vep))])
    TCGA_nodels_df["strand"] = TCGA_nodels_df["strand"].fillna(1).astype(float)

embeddings = []
average_embeddings = []

tokenizer_fn = functools.partial(
    tokenizer.encode_plus,
    max_length=512,
    padding="max_length",
    truncation=True,
    return_tensors="pt",
)

chunk = 1024
batch_size = 16
for i in tqdm(range(len(TCGA_nodels_df)//chunk), desc="Dataframe list", leave=True):
# for i in tqdm(range(5), desc="Dataframe list", leave=True):
    df = TCGA_nodels_df.iloc[i*chunk:(i+1)*chunk, :].copy()

    if args.input_vep:
        df['ref_seq'], df['mut_seq'], df['mutant_index'] = zip(*df.apply(lambda x: execute_kmers(REFERENCE, seq_length, x['chrom'], x['start'], x['ref'], x['alt'], logger, x['strand']), axis=1))
    else:
        df['ref_seq'], df['mut_seq'], df['mutant_index'] = zip(*df.apply(lambda x: execute_kmers(REFERENCE, seq_length, x['chrom'], x['start'], x['ref'], x['alt'], logger), axis=1))

    if i == 0:
        df.to_csv(args.output + '.tsv.gz', compression="gzip", sep="\t", index=False)
    else:
        df.to_csv(args.output + '.tsv.gz', compression="gzip", sep="\t", index=False,
                  header=False,
                  mode='a')

    sequences = [[x, y] for x, y in zip(df['ref_seq'], df['mut_seq'])]
    dataset = SequenceDataset(sequences, 512, tokenizer, verbose=False)
    dataloader = DataLoader(dataset, batch_size=16)

    siamese_bert.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Dataloader", leave=False):
            batch = {k: v.to("cuda") for k, v in batch.items()}
            outputs = siamese_bert(batch["input_ids"], batch["attention_mask"])
            embeddings.append(outputs[3].detach().cpu().numpy())
            average_embeddings.append(outputs[4].detach().cpu().numpy())

# Concatenate embeddings into an array of shape (num_sequences, hidden_size)
embeddings = np.concatenate(embeddings)
average_embeddings = np.concatenate(average_embeddings)

np.save(args.output + ".embedding_CLS.npy", embeddings, allow_pickle=False)
np.save(args.output + ".embedding_average.npy", average_embeddings, allow_pickle=False)
