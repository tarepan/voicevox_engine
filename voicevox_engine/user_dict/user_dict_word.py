"""ユーザー辞書を構成する言葉（単語）関連の処理"""

from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel, Field

from voicevox_engine.user_dict.model import (
    USER_DICT_MAX_PRIORITY,
    USER_DICT_MIN_PRIORITY,
    UserDictWord,
    WordTypes,
)


class _PartOfSpeechDetail(BaseModel):
    """
    品詞ごとの情報
    """

    part_of_speech: str = Field(title="品詞")
    part_of_speech_detail_1: str = Field(title="品詞細分類1")
    part_of_speech_detail_2: str = Field(title="品詞細分類2")
    part_of_speech_detail_3: str = Field(title="品詞細分類3")
    # context_idは辞書の左・右文脈IDのこと
    # https://github.com/VOICEVOX/open_jtalk/blob/427cfd761b78efb6094bea3c5bb8c968f0d711ab/src/mecab-naist-jdic/_left-id.def # noqa
    context_id: int = Field(title="文脈ID")
    cost_candidates: list[int] = Field(title="コストのパーセンタイル")
    accent_associative_rules: list[str] = Field(title="アクセント結合規則の一覧")


_costs_proper_noun = [-988, 3488, 4768, 6048, 7328, 8609, 8734, 8859, 8984, 9110, 14176]
_costs_common_noun = [-4445, 49, 1473, 2897, 4321, 5746, 6554, 7362, 8170, 8979, 15001]
_costs_verb = [3100, 6160, 6360, 6561, 6761, 6962, 7414, 7866, 8318, 8771, 13433]
_costs_adjective = [1527, 3266, 3561, 3857, 4153, 4449, 5149, 5849, 6549, 7250, 10001]
_costs_suffix = [4399, 5373, 6041, 6710, 7378, 8047, 9440, 10834, 12228, 13622, 15847]


part_of_speech_data: dict[WordTypes, _PartOfSpeechDetail] = {
    WordTypes.PROPER_NOUN: _PartOfSpeechDetail(
        part_of_speech="名詞",
        part_of_speech_detail_1="固有名詞",
        part_of_speech_detail_2="一般",
        part_of_speech_detail_3="*",
        context_id=1348,
        cost_candidates=_costs_proper_noun,
        accent_associative_rules=["*", "C1", "C2", "C3", "C4", "C5"],
    ),
    WordTypes.COMMON_NOUN: _PartOfSpeechDetail(
        part_of_speech="名詞",
        part_of_speech_detail_1="一般",
        part_of_speech_detail_2="*",
        part_of_speech_detail_3="*",
        context_id=1345,
        cost_candidates=_costs_common_noun,
        accent_associative_rules=["*", "C1", "C2", "C3", "C4", "C5"],
    ),
    WordTypes.VERB: _PartOfSpeechDetail(
        part_of_speech="動詞",
        part_of_speech_detail_1="自立",
        part_of_speech_detail_2="*",
        part_of_speech_detail_3="*",
        context_id=642,
        cost_candidates=_costs_verb,
        accent_associative_rules=["*"],
    ),
    WordTypes.ADJECTIVE: _PartOfSpeechDetail(
        part_of_speech="形容詞",
        part_of_speech_detail_1="自立",
        part_of_speech_detail_2="*",
        part_of_speech_detail_3="*",
        context_id=20,
        cost_candidates=_costs_adjective,
        accent_associative_rules=["*"],
    ),
    WordTypes.SUFFIX: _PartOfSpeechDetail(
        part_of_speech="名詞",
        part_of_speech_detail_1="接尾",
        part_of_speech_detail_2="一般",
        part_of_speech_detail_3="*",
        context_id=1358,
        cost_candidates=_costs_suffix,
        accent_associative_rules=["*", "C1", "C2", "C3", "C4", "C5"],
    ),
}


@dataclass
class WordProperty:
    """単語属性のあつまり"""

    surface: str  # 単語情報
    pronunciation: str  # 単語情報
    accent_type: int  # 単語情報
    word_type: WordTypes | None = None  # 品詞
    priority: int | None = None  # 優先度


def create_word(word_property: WordProperty) -> UserDictWord:
    """単語オブジェクトを生成する。"""
    word_type: WordTypes | None = word_property.word_type
    if word_type is None:
        word_type = WordTypes.PROPER_NOUN
    if word_type not in part_of_speech_data.keys():
        raise UserDictInputError("不明な品詞です")

    priority: int | None = word_property.priority
    if priority is None:
        priority = 5
    if not USER_DICT_MIN_PRIORITY <= priority <= USER_DICT_MAX_PRIORITY:
        raise UserDictInputError("優先度の値が無効です")

    pos_detail = part_of_speech_data[word_type]
    return UserDictWord(
        surface=word_property.surface,
        context_id=pos_detail.context_id,
        priority=priority,
        part_of_speech=pos_detail.part_of_speech,
        part_of_speech_detail_1=pos_detail.part_of_speech_detail_1,
        part_of_speech_detail_2=pos_detail.part_of_speech_detail_2,
        part_of_speech_detail_3=pos_detail.part_of_speech_detail_3,
        inflectional_type="*",
        inflectional_form="*",
        stem="*",
        yomi=word_property.pronunciation,
        pronunciation=word_property.pronunciation,
        accent_type=word_property.accent_type,
        mora_count=None,
        accent_associative_rule="*",
    )


class UserDictInputError(Exception):
    """受け入れ不可能な入力値に起因するエラー"""

    pass


def _search_cost_candidates(context_id: int) -> list[int]:
    for value in part_of_speech_data.values():
        if value.context_id == context_id:
            return value.cost_candidates
    raise UserDictInputError("品詞IDが不正です")


def cost2priority(context_id: int, cost: int) -> int:
    assert -32768 <= cost <= 32767
    cost_candidates = _search_cost_candidates(context_id)
    # cost_candidatesの中にある値で最も近い値を元にpriorityを返す
    # 参考: https://qiita.com/Krypf/items/2eada91c37161d17621d
    # この関数とpriority2cost関数によって、辞書ファイルのcostを操作しても最も近いpriorityのcostに上書きされる
    return (
        USER_DICT_MAX_PRIORITY
        - np.argmin(np.abs(np.array(cost_candidates) - cost)).item()
    )


def priority2cost(context_id: int, priority: int) -> int:
    assert USER_DICT_MIN_PRIORITY <= priority <= USER_DICT_MAX_PRIORITY
    cost_candidates = _search_cost_candidates(context_id)
    return cost_candidates[USER_DICT_MAX_PRIORITY - priority]