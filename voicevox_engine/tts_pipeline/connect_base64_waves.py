"""音声データの結合"""

import base64
import io

import numpy as np
import soundfile
from numpy.typing import NDArray
from soxr import resample


class ConnectBase64WavesException(Exception):
    """Base64 エンコードされた音声波形の結合に失敗した。"""

    def __init__(self, message: str):
        self.message = message


def decode_base64_waves(waves: list[str]) -> list[tuple[NDArray[np.float64], int]]:
    """
    base64エンコードされた複数のwavデータをデコードする

    Parameters
    ----------
    waves: list[str]
        base64エンコードされたwavデータのリスト

    Returns
    -------
    waves_nparray_sr: list[tuple[NDArray[np.float64], int]]
        (NumPy配列の音声波形データ, サンプリングレート) 形式のタプルのリスト
    """
    if len(waves) == 0:
        raise ConnectBase64WavesException("wavファイルが含まれていません")

    waves_nparray_sr = []
    for wave in waves:
        try:
            wav_bin = base64.standard_b64decode(wave)
        except ValueError as e:
            raise ConnectBase64WavesException("base64デコードに失敗しました") from e
        try:
            _data = soundfile.read(io.BytesIO(wav_bin))
        except Exception as e:
            raise ConnectBase64WavesException(
                "wavファイルを読み込めませんでした"
            ) from e
        waves_nparray_sr.append(_data)

    return waves_nparray_sr


def connect_base64_waves(waves: list[str]) -> tuple[NDArray[np.float64], int]:
    """複数の base64 エンコードされた音声波形を1つに結合する。"""
    waves_nparray_sr = decode_base64_waves(waves)

    max_sampling_rate = max([sr for _, sr in waves_nparray_sr])
    max_channels = max([x.ndim for x, _ in waves_nparray_sr])
    assert 0 < max_channels <= 2

    waves_nparray_list = []
    for nparray, sr in waves_nparray_sr:
        if sr != max_sampling_rate:
            nparray = resample(nparray, sr, max_sampling_rate)
        if nparray.ndim < max_channels:
            nparray = np.array([nparray, nparray]).T
        waves_nparray_list.append(nparray)

    return np.concatenate(waves_nparray_list), max_sampling_rate
