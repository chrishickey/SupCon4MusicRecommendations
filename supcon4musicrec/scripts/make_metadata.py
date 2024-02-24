#!/usr/bin/python
import argparse
import csv
import json
import os.path

from tqdm import tqdm


def parse_arguments() -> argparse.Namespace:
    # setup arg parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "raw_track_file",
        type=str,
        help="path to raw_track.csv (found here https://os.unil.cloud.switch.ch/fma/fma_metadata.zip)",
    )
    parser.add_argument(
        "outdir",
        type=str,
        help="Directory to output files",
    )
    return parser.parse_args()


def main(
    raw_track_file: str,
    outdir: str,
):
    os.makedirs(outdir, exist_ok=True)
    with open(raw_track_file) as csvfile:
        reader = csv.DictReader(csvfile)
        all_rows = [row for row in reader]
        for row in tqdm(all_rows, total=len(all_rows)):
            file_path = os.path.join(outdir, f"{row['track_id']}.json")
            with open(file_path, "w+") as fh:
                json.dump(row, fh, indent=4)


if __name__ == "__main__":
    args = parse_arguments()
    main(
        args.raw_track_file,
        args.outdir,
    )
