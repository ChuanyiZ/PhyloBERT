import argparse
import random
import numpy as np
import pysam

def cut_no_overlap(length, kmer=1, max_prob=0.5):
    cuts = []
    while length:
        if length <= 509+kmer:
            cuts.append(length)
            break
        else:
            if random.random() > max_prob:
                cut = max(int(random.random()*(509+kmer)), 5)
            else:
                cut = 509+kmer
            cuts.append(cut)
            length -= cut

    return cuts


def sampling(length, kmer=1, sampling_rate=1):
    times = int(length*sampling_rate/256)
    starts = []
    ends = []
    for i in range(times):
        cut = max(int(random.random()*(509+kmer)), 5)
        start = np.random.randint(length-kmer)
        starts.append(start)
        ends.append(start+cut)
    
    return starts, ends


def sampling_fix(length, kmer=1, sampling_rate=1, fix_length=10245):
    times = int(length*sampling_rate/fix_length)
    starts = []
    ends = []
    for i in range(times):
        cut = fix_length
        start = np.random.randint(length-6-fix_length)
        starts.append(start)
        ends.append(start+cut)
    
    return starts, ends


def get_kmer_sentence(original_string, kmer=1, stride=1):
    if kmer == -1:
        return original_string

    sentence = ""
    original_string = original_string.replace("\n", "")
    i = 0
    while i < len(original_string)-kmer:
        sentence += original_string[i:i+kmer] + " "
        i += stride
    
    return sentence[:-1].strip("\"")



def get_kmer_sequence(original_string, kmer=1):
    if kmer == -1:
        return original_string

    sequence = []
    original_string = original_string.replace("\n", "")
    for i in range(len(original_string)-kmer):
        sequence.append(original_string[i:i+kmer])
    
    sequence.append(original_string[-kmer:])
    return sequence

def Process(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    ref = pysam.FastaFile(args.file_path)
    if args.output_path == None:
        args.output_path = args.file_path

    if args.sampling_rate!=1.0:
        new_file_path = f"{args.output_path}_{args.chrom}_sam{str(args.kmer)}"
    else:
        new_file_path = f"{args.output_path}_{args.chrom}_cut{str(args.kmer)}"
    with open(new_file_path, "w") as new_file:
        chromosome = ref.fetch(args.chrom)
        print(args.chrom, len(chromosome))
        line_length = len(chromosome)
        if args.sampling_rate != 1.0:
            starts, ends = sampling_fix(length=line_length, kmer=args.kmer, sampling_rate=args.sampling_rate, fix_length=args.length)
            for i in range(len(starts)):
                new_line = chromosome[starts[i]:ends[i]].upper()
                if new_line.count('N') == 0:
                    sentence = get_kmer_sentence(new_line, kmer=args.kmer)
                    new_file.write(sentence + "\n")
            
        else:
            cuts = cut_no_overlap(length=line_length, kmer=args.kmer)
            start = 0
            for cut in cuts:
                new_line = chromosome[start:start+cut].upper()
                if new_line.count('N') == 0:
                    sentence = get_kmer_sentence(new_line, kmer=args.kmer)
                    new_file.write(sentence + "\n")
                start += cut


def main():
    random.seed(42)
    np.random.seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sampling_rate", 
        default=1.0,
        type=float,
        help="We will sample sampling_rate*total_length*2/512 times",
    )
    parser.add_argument(
        "--kmer",
        default=6,
        type=int,
        help="K-mer",
    )
    parser.add_argument(
        "--length",
        default=10000,
        type=int,
        help="Length of the sampled sequence",
    )
    parser.add_argument(
        "--file_path",
        default=None,
        type=str,
        help="The path of the file to be processed",
    )
    parser.add_argument(
        "--output_path",
        default=None,
        type=str,
        help="The path of the processed data",
    )
    args = parser.parse_args()

    chroms = range(22,23)
    for idx, chrom in enumerate(chroms):
        args.chrom = f"chr{chrom}"
        args.seed = idx
        Process(args)


if __name__ == "__main__":
    main()
