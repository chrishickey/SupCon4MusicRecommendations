from multiprocessing import Pool

from tqdm import tqdm


def multiprocess_transform(function, iterable, n_workers: int = 2):
    """
    Apply a transformation in a multiprocess way.

    Args:
        function: transformation to apply on dataste
        iterable: iterable on which to apply the transformation
        n_workers (int): number of workers for the multithreading

    Example:
        >>> melextractor = MelspectrogramExtractor()
        >>> melextractor('src_folder_path', 'dest_folder_path', n_workers=2)
    """
    if n_workers == 1:
        iterable = tqdm(iterable, total=len(iterable))
        return list(map(function, iterable))

    else:
        with Pool(processes=n_workers) as pool:
            with tqdm(total=len(iterable)) as pbar:
                outputs = []
                for output in pool.imap_unordered(function, iterable):
                    outputs.append(output)
                    pbar.update()
            return outputs
