#!/usr/bin/python
import argparse
import multiprocessing as mp

from loguru import logger

from supcon4musicrec.utils.preprocessors import (
    BPMExtractor,
    DecibelMelspectrogramExtractor,
)

N_WORKERS = 2


def parse_arguments() -> argparse.Namespace:
    # setup arg parser
    parser = argparse.ArgumentParser()
    parser.add_argument("mp3_dir", help="location where mp3 files are stored")
    parser.add_argument("mel_outdir", help="location to write mel output")
    parser.add_argument("json_outdir", help="location to write json output")
    parser.add_argument(
        "--workers", type=int, default=N_WORKERS, help="number of workers doing the job"
    )
    return parser.parse_args()


def main(
    mp3_dir: str,
    mel_outdir: str,
    json_outdir: str,
    workers: int,
):
    logger.info(
        f"# 1 PREPROCESSING SONGS:\t"
        f"Saving songs from {mp3_dir} as decibel melspectrograms to {mel_outdir}."
    )
    DecibelMelspectrogramExtractor(
        dest_folder_path=mel_outdir,
        overwrite=False,  # This value is needed to speed up retries if there is failure
    )(src_folder_path=mp3_dir, n_workers=workers)

    logger.info(
        f"# 2 CALCULATING BPM:\t" f"Saving song bpms from {mp3_dir} to {json_outdir}."
    )
    BPMExtractor(
        dest_folder_path=json_outdir,
        overwrite=False,  # This value is needed to speed up retries if there is failure
    )(src_folder_path=mp3_dir, n_workers=workers)

    return True


if __name__ == "__main__":
    mp.set_start_method("spawn")
    args = parse_arguments()
    main(args.mp3_dir, args.mel_outdir, args.json_outdir, args.workers)
