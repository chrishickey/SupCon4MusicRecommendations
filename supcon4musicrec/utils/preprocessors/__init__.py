import json
import os
import typing

import librosa
import numpy as np

from supcon4musicrec.utils.multiprocessing import multiprocess_transform


class BaseAudioExtractor:
    """
    A callable class that converts all MP3 audio files contained in a folder to some feature. Allows multiprocessing.
    """

    def __init__(
        self,
        *,
        dest_folder_path: typing.Optional[str] = None,
        sample_rate: typing.Optional[int] = None,
        overwrite: bool = True,
        **kwargs,
    ) -> None:
        """
        Initialiser for the MelspectrogramExtractor class.

        Args:
            dest_folder_path (str): path to the folder where to save resulting melspectrograms.
            sample_rate (int, optional): The sample rate for the audio files.
        """
        self.dest_folder_path = dest_folder_path
        self.sample_rate = sample_rate
        self.kwargs = kwargs
        self.overwrite = overwrite

        if self.dest_folder_path and not os.path.isdir(dest_folder_path):
            os.makedirs(dest_folder_path)

    def get_filename(self, file_name: str) -> str:
        return os.path.join(
            self.dest_folder_path,
            f"{os.path.splitext(os.path.basename(file_name))[0]}.npy",
        )

    def read(self, file_name: str) -> tuple[np.ndarray, float]:
        """
        Transforms a MP3 file to a numpy array.

        Args:
            file_name (str): Name of the audio file to read.

        Returns:
            tuple: The audio data as a numpy array and the sample rate.

        """
        return librosa.load(file_name, sr=self.sample_rate, mono=False)

    def do_not_overwrite(self, file_name) -> bool:
        return not self.overwrite and os.path.exists(self.get_filename(file_name))

    def extract(self, file_name: str) -> np.ndarray:
        raise NotImplementedError("Implement in subclass.")

    def save(self, file_name: str, data: np.ndarray):
        """
        Saves the numpy array to a .npy file.

        Args:
            file_name (str): name of the file where to save audio data.
            data (np.ndarray): audio data to save.
        """
        np.save(self.get_filename(file_name), data)

    def __call__(
        self,
        src_folder_path: str,
        n_workers: typing.Optional[int] = None,
    ):
        """
        Converts all files contained in a folder to melspectrograms. Allows multiprocessing.

        Args:
            src_folder_path (str): path to the folder containing MP3 audio files.
            n_workers (int, Optional): number of worker processes to use.
        """
        assert (
            self.dest_folder_path is not None
        ), "Must specify destination folder to save files to"
        mp3_files = [
            os.path.join(src_folder_path, file_name)
            for file_name in os.listdir(src_folder_path)
            if os.path.splitext(file_name)[1] == ".mp3"
        ]  # consider only MP3 files
        assert (
            n_workers is None or n_workers > 0
        ), "Valid values are None or a number greater than 0"
        multiprocess_transform(self.extract, mp3_files, n_workers)


class MelspectrogramExtractor(BaseAudioExtractor):
    def extract(self, file_name: str) -> np.ndarray:
        """
        Computes the melspectrogram of an input audio file using Librosa
        and saves the resulting numpy array.

        Args:
            file_name (str): name of the audio file to read
        """
        y, sr = self.read(file_name)
        Y = librosa.feature.melspectrogram(y=y, sr=sr, **self.kwargs)
        if self.dest_folder_path:
            self.save(file_name, Y)
        return Y


class DecibelMelspectrogramExtractor(BaseAudioExtractor):
    def __init__(self, *, delete_after_preprocessing: bool = False, **kwargs):
        """
        Initializes the DecibelMelspectrogramExtractor class.

        Args:
            sample_rate (int, optional): The sample rate for the audio files. Defaults to DEFAULT_SAMPLE_RATE.
        """
        self.delete_after_preprocessing = delete_after_preprocessing
        # sample rate for all files.
        super().__init__(**kwargs)

    def extract(self, file_name: str) -> typing.Union[None, np.ndarray]:
        """
        Computes the melspectrogram of an input audio file using Librosa
        and saves the resulting numpy array.

        Args:
            file_name (str): name of the audio file to read

        Notes:
            Original source code can be found here
            https://github.com/eldrin/MTLMusicRepresentation-PyTorch/blob/master/musmtl/utils.py#L61
        """
        if self.do_not_overwrite(file_name):
            return

        y, sr = self.read(file_name)

        if y.ndim == 1:
            y = np.vstack([y, y])

        Y = librosa.power_to_db(
            np.array(
                [
                    librosa.feature.melspectrogram(y=ch, sr=sr, **self.kwargs).T
                    for ch in y
                ]
            )
        ).astype(
            np.float32
        )  # (2, t, 128) or (1, t, 128) if mono

        if self.dest_folder_path:
            self.save(file_name, Y)

        if self.delete_after_preprocessing:
            os.remove(file_name)
        return Y


class BaseAudioJsonExtractor(BaseAudioExtractor):
    def save(self, file_name: str, data: typing.Dict[str, typing.Any]):
        """Save mp3 metadata"""
        with open(self.get_filename(file_name), "w+") as fh:
            json.dump(data, fh, indent=4)

    def get_filename(self, file_name: str):
        """Get json file name for mp3 file."""
        return os.path.join(
            self.dest_folder_path, os.path.basename(file_name).replace(".mp3", ".json")
        )


class BPMExtractor(BaseAudioJsonExtractor):
    def extract(self, file_name: str) -> typing.Union[None, float]:
        """
        Computes the bpm of an input audio file using Librosa
        and saves the resulting numpy array.

        Args:
            file_name (str): name of the audio file to read
        """
        if self.do_not_overwrite(file_name):
            return

        y, sr = self.read(file_name)
        bpm = librosa.beat.tempo(y=y, sr=sr)
        bpm = round(np.average(bpm))

        if self.dest_folder_path:
            self.save(file_name, {"bpm": bpm})
        return bpm
