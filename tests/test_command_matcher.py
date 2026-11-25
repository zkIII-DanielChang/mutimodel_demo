from src.asr.command_matcher import CommandMatcher, _char_bigrams, _cosine


def test_char_bigrams_basic():
    g = _char_bigrams("打开一号管")
    assert "打开" in g
    assert "一号" in g


def test_cosine_identical():
    g = _char_bigrams("打开一号管")
    assert abs(_cosine(g, g) - 1.0) < 1e-6


def test_cosine_unrelated():
    a = _char_bigrams("打开一号管")
    b = _char_bigrams("今天天气不错")
    assert _cosine(a, b) < 0.3


def test_match_exact():
    m = CommandMatcher(["打开一号管", "关闭一号管"])
    r = m.match("打开一号管", threshold=0.5)
    assert r["matched"] is True
    assert r["command"] == "打开一号管"


def test_match_with_extra_words():
    m = CommandMatcher(["打开一号管", "关闭一号管"])
    r = m.match("请打开一号管", threshold=0.5)
    assert r["matched"] is True
    assert r["command"] == "打开一号管"


def test_no_match_below_threshold():
    m = CommandMatcher(["打开一号管", "关闭一号管"])
    r = m.match("今天天气不错", threshold=0.9)
    assert r["matched"] is False
    assert r["command"] is None