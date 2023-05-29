import csv
import argparse

def kmer_to_sequence(kmers: str, sep=' '):
    k = kmers.find(sep)
    return kmers[:k] + ''.join(kmer[-1] for kmer in kmers.split(sep))

def main():
    parser = argparse.ArgumentParser(description="Convert k-mer dataset TSV file to sequences")
    parser.add_argument('input', help='input file')
    parser.add_argument('output', help='output file')
    args = parser.parse_args()

    with open(args.input) as ifile, open(args.output, 'wt') as ofile:
        reader = csv.DictReader(ifile, delimiter='\t')
        writer = csv.DictWriter(ofile, fieldnames=reader.fieldnames, delimiter='\t')
        writer.writeheader()
        for row in reader:
            writer.writerow({
                "seq_ref": kmer_to_sequence(row["seq_ref"]),
                "seq_mut": kmer_to_sequence(row["seq_mut"]),
                "label": row["label"],
            })

if __name__ == "__main__":
    main()
