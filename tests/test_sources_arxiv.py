from core.sources import arxiv_source


def test_build_query_no_year_filter():
    assert arxiv_source.build_query("transformers", None, None) == "transformers"


def test_build_query_with_year_range():
    q = arxiv_source.build_query("transformers", 2020, 2022)
    assert q == "(transformers) AND submittedDate:[202001010000 TO 202212312359]"


def test_build_query_with_only_year_from():
    q = arxiv_source.build_query("transformers", 2020, None)
    assert q == "(transformers) AND submittedDate:[202001010000 TO 210012312359]"


def test_build_query_with_only_year_to():
    q = arxiv_source.build_query("transformers", None, 2022)
    assert q == "(transformers) AND submittedDate:[199001010000 TO 202212312359]"
