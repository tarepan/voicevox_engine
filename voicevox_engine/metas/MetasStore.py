import json
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Tuple

from voicevox_engine.metas.Metas import CoreSpeaker, EngineSpeaker, Speaker, StyleInfo

if TYPE_CHECKING:
    from voicevox_engine.synthesis_engine.synthesis_engine_base import (
        SynthesisEngineBase,
    )


class MetasStore:
    """
    話者やスタイルのメタ情報を管理する
    """

    def __init__(self, engine_speakers_path: Path) -> None:
        """
        Parameters
        ----------
        engine_speakers_path : Path
            エンジンに含まれる話者メタ情報ディレクトリのパス。
        """
        # エンジンに含まれる各話者のメタ情報
        self._loaded_metas: Dict[str, EngineSpeaker] = {
            folder.name: EngineSpeaker(
                **json.loads((folder / "metas.json").read_text(encoding="utf-8"))
            )
            for folder in engine_speakers_path.iterdir()
        }

    def _speaker_engine_metas(self, speaker_uuid: str) -> EngineSpeaker:
        """
        エンジンに含まれる指定話者のメタ情報を取得
        Parameters
        ----------
        speaker_uuid : str
            話者UUID
        Returns
        -------
        ret : EngineSpeaker
            エンジンに含まれる指定話者のメタ情報
        """
        return self._loaded_metas[speaker_uuid]

    def _combine_metas(self, core_metas: List[CoreSpeaker]) -> List[Speaker]:
        """
        コアに含まれる話者メタ情報と指定エンジンに含まれる話者コア情報を統合
        Parameters
        ----------
        core_metas : List[CoreSpeaker]
            コアに含まれる話者メタ情報
        Returns
        -------
        ret : List[Speaker]
            エンジンとコアに含まれる話者メタ情報
        """
        # 話者単位でエンジン・コアに含まれるメタ情報を統合
        return [
            Speaker(
                **self._speaker_engine_metas(speaker_meta.speaker_uuid).dict(),
                **speaker_meta.dict(),
            )
            for speaker_meta in core_metas
        ]

    # FIXME: engineではなくList[CoreSpeaker]を渡す形にすることで
    # SynthesisEngineBaseによる循環importを修正する
    def load_combined_metas(self, engine: "SynthesisEngineBase") -> List[Speaker]:
        """
        コアに含まれる話者メタ情報に、指定エンジンに含まれる話者メタ情報を付加
        Parameters
        ----------
        engine : SynthesisEngineBase
            指定されたエンジン
        Returns
        -------
        ret : List[Speaker]
            エンジンとコアに含まれる話者メタ情報
        """
        # コアに含まれる話者メタ情報の収集
        core_metas = [CoreSpeaker(**speaker) for speaker in json.loads(engine.speakers)]
        # エンジンに含まれる話者メタ情報との統合
        return self._combine_metas(core_metas)


def construct_lookup(speakers: List[Speaker]) -> Dict[int, Tuple[Speaker, StyleInfo]]:
    """
    スタイルID に話者メタ情報・スタイルメタ情報を紐付ける対応表を生成
    Parameters
    ----------
    speakers : List[Speaker]
        話者メタ情報
    Returns
    -------
    ret : Dict[int, Tuple[Speaker, StyleInfo]]
        スタイルID に話者メタ情報・スタイルメタ情報が紐付いた対応表
    """
    lookup_table: Dict[int, Tuple[Speaker, StyleInfo]] = dict()
    for speaker in speakers:
        for style in speaker.styles:
            lookup_table[style.id] = (speaker, style)
    return lookup_table
