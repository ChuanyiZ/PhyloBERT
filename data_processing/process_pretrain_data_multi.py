from multiprocessing import Pool
import copy
import argparse

from process_pretrain_data import Process

# filenames = ['xaa', 'xab', 'xac', 'xad', 'xae', 'xaf', 'xag', 'xah', 'xai', 'xaj', 'xak', 'xal', 'xam', 'xan', 'xao', 'xap', 'xaq', 'xar', 'xas', 'xat', 'xau', 'xav', 'xaw']
# filenames = ['xaa', 'xab']

def main():

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
        required=True,
        type=str,
        help="The path of the file to be processed",
    )
    parser.add_argument(
        "--output_path",
        required=True,
        type=str,
        help="The path of the file to be processed",
    )
    parser.add_argument(
        "-c",
        default=1,
        type=int,
        help="number of cores to use",
    )
    args = parser.parse_args()

    # multiprocess
    p = Pool(args.c)

    chroms = list(range(1,23)) + ["X", "Y"]
    for idx, chrom in enumerate(chroms):
        arg_new = copy.deepcopy(args)
        arg_new.chrom = f"chr{chrom}"
        arg_new.seed = idx
        # arg_new.file_path = arg_new.output_path + filename
        p.apply_async(Process, args=(arg_new,))
    
    p.close()
    p.join()




if __name__ == "__main__":
  main()
