import re
from .schema import Chunk


def validate_citations(
    answer: str,
    results: list[tuple[Chunk, float]],
) -> bool:
    # 提取引用编号，例如 "[1] and [2]" -> [1, 2]
    citations = [int(x) for x in re.findall(r"\[(\d+)\]", answer)]

    # TODO 1: 没有引用时如何处理？
    if not citations:
        raise ValueError("invalid citation")
    # TODO 2: 判断所有编号是否合法
    # 有效范围：1 ~ len(results)
    if all([1 <= citation <= len(results) for citation in citations]):
        return True
    else:
        return False

    # TODO 3: 返回 bool
