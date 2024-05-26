from dataclasses import dataclass
from enum import Enum

MIN_PRIORITY = 0
MAX_PRIORITY = 10


@dataclass(frozen=True)
class _PartOfSpeechDetail:
    """品詞ごとの情報"""

    part_of_speech: str  # 品詞
    part_of_speech_detail_1: str  # 品詞細分類1
    part_of_speech_detail_2: str  # 品詞細分類2
    part_of_speech_detail_3: str  # 品詞細分類3
    context_id: int  # 文脈ID（辞書の左・右文脈ID）https://github.com/VOICEVOX/open_jtalk/blob/427cfd761b78efb6094bea3c5bb8c968f0d711ab/src/mecab-naist-jdic/_left-id.def # noqa
    cost_candidates: list[int]  # コストのパーセンタイル
    accent_associative_rules: list[str]  # アクセント結合規則の一覧


class WordTypes(str, Enum):
    """言葉の種類。PROPER_NOUN（固有名詞）、COMMON_NOUN（普通名詞）、VERB（動詞）、ADJECTIVE（形容詞）、SUFFIX（語尾）のいずれか。"""

    PROPER_NOUN = "PROPER_NOUN"
    COMMON_NOUN = "COMMON_NOUN"
    VERB = "VERB"
    ADJECTIVE = "ADJECTIVE"
    SUFFIX = "SUFFIX"


part_of_speech_data: dict[WordTypes, _PartOfSpeechDetail] = {
    WordTypes.PROPER_NOUN: _PartOfSpeechDetail(
        part_of_speech="名詞",
        part_of_speech_detail_1="固有名詞",
        part_of_speech_detail_2="一般",
        part_of_speech_detail_3="*",
        context_id=1348,
        cost_candidates=[
            -988,
            3488,
            4768,
            6048,
            7328,
            8609,
            8734,
            8859,
            8984,
            9110,
            14176,
        ],
        accent_associative_rules=[
            "*",
            "C1",
            "C2",
            "C3",
            "C4",
            "C5",
        ],
    ),
    WordTypes.COMMON_NOUN: _PartOfSpeechDetail(
        part_of_speech="名詞",
        part_of_speech_detail_1="一般",
        part_of_speech_detail_2="*",
        part_of_speech_detail_3="*",
        context_id=1345,
        cost_candidates=[
            -4445,
            49,
            1473,
            2897,
            4321,
            5746,
            6554,
            7362,
            8170,
            8979,
            15001,
        ],
        accent_associative_rules=[
            "*",
            "C1",
            "C2",
            "C3",
            "C4",
            "C5",
        ],
    ),
    WordTypes.VERB: _PartOfSpeechDetail(
        part_of_speech="動詞",
        part_of_speech_detail_1="自立",
        part_of_speech_detail_2="*",
        part_of_speech_detail_3="*",
        context_id=642,
        cost_candidates=[
            3100,
            6160,
            6360,
            6561,
            6761,
            6962,
            7414,
            7866,
            8318,
            8771,
            13433,
        ],
        accent_associative_rules=[
            "*",
        ],
    ),
    WordTypes.ADJECTIVE: _PartOfSpeechDetail(
        part_of_speech="形容詞",
        part_of_speech_detail_1="自立",
        part_of_speech_detail_2="*",
        part_of_speech_detail_3="*",
        context_id=20,
        cost_candidates=[
            1527,
            3266,
            3561,
            3857,
            4153,
            4449,
            5149,
            5849,
            6549,
            7250,
            10001,
        ],
        accent_associative_rules=[
            "*",
        ],
    ),
    WordTypes.SUFFIX: _PartOfSpeechDetail(
        part_of_speech="名詞",
        part_of_speech_detail_1="接尾",
        part_of_speech_detail_2="一般",
        part_of_speech_detail_3="*",
        context_id=1358,
        cost_candidates=[
            4399,
            5373,
            6041,
            6710,
            7378,
            8047,
            9440,
            10834,
            12228,
            13622,
            15847,
        ],
        accent_associative_rules=[
            "*",
            "C1",
            "C2",
            "C3",
            "C4",
            "C5",
        ],
    ),
}
