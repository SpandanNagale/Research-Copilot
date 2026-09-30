import random
import re

from core.llm import ThinkFilter, strip_think


def _reference_strip(s: str) -> str:
    open_idx = s.find("<think>")
    close_idx = s.find("</think>")
    if close_idx != -1 and (open_idx == -1 or close_idx < open_idx):
        s = s[close_idx + len("</think>"):]
    return re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL)


def _feed_in_chunks(text: str, chunk_sizes: list[int]) -> str:
    filt = ThinkFilter()
    out = []
    i = 0
    for size in chunk_sizes:
        if i >= len(text):
            break
        out.append(filt.feed(text[i : i + size]))
        i += size
    if i < len(text):
        out.append(filt.feed(text[i:]))
    out.append(filt.finish())
    return "".join(out)


def test_strip_think_basic_block():
    assert strip_think("before<think>hidden reasoning</think>after") == "beforeafter"


def test_strip_think_leading_orphan_close():
    assert strip_think("stray reasoning</think>visible answer") == "visible answer"


def test_strip_think_no_tags_passes_through():
    assert strip_think("just a normal answer") == "just a normal answer"


def test_strip_think_literal_less_than_not_swallowed():
    assert strip_think("a < b and c > d") == "a < b and c > d"


def test_strip_think_multiple_blocks():
    text = "a<think>x</think>b<think>y</think>c"
    assert strip_think(text) == "abc"


def test_streaming_filter_splits_tag_across_chunks():
    filt = ThinkFilter()
    out = []
    out.append(filt.feed("before<thi"))
    out.append(filt.feed("nk>hidden</th"))
    out.append(filt.feed("ink>after"))
    out.append(filt.finish())
    assert "".join(out) == "beforeafter"


def test_streaming_filter_dangling_partial_tag_at_end_is_flushed():
    filt = ThinkFilter()
    out = [filt.feed("hello <thi")]
    out.append(filt.finish())  # never completes -> must be flushed as literal text
    assert "".join(out) == "hello <thi"


def test_streaming_filter_unclosed_think_block_drops_content():
    filt = ThinkFilter()
    out = [filt.feed("before<think>never closes")]
    out.append(filt.finish())
    assert "".join(out) == "before"


def _random_text(rng: random.Random) -> str:
    parts = []
    if rng.random() < 0.15:
        parts.append("stray leading reasoning</think>")
    for _ in range(rng.randint(0, 3)):
        parts.append(rng.choice(["plain text ", "a < b ", "c > d ", "x<yz ", "normal words "]))
        if rng.random() < 0.5:
            parts.append(f"<think>{'reasoning ' * rng.randint(1, 5)}</think>")
    parts.append(rng.choice(["", "trailing text", "final < bit"]))
    return "".join(parts)


def test_fuzz_random_chunk_sizes_match_reference():
    rng = random.Random(1234)
    for _ in range(300):
        text = _random_text(rng)
        chunk_sizes = [rng.randint(1, 7) for _ in range(len(text) + 1)]
        streamed = _feed_in_chunks(text, chunk_sizes)
        assert streamed == _reference_strip(text), f"mismatch for {text!r} with chunks {chunk_sizes!r}"
