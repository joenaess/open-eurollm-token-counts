import pytest
from src.analyze import extract_language, get_source_type


def test_extract_language():
    # Explicit language
    assert extract_language("some_dataset", "eng_Latn") == "eng_Latn"
    assert extract_language("some_dataset", "bos_Latn-eng_Latn") == "bos_Latn"
    assert extract_language("some_dataset", "eng_Latn-als_Latn") == "als_Latn"
    assert extract_language("some_dataset", "clean/als_Latn") == "als_Latn"
    assert (
        extract_language("nemotron-cc-tower", "parallel/tower72b/deu_Latn")
        == "deu_Latn"
    )

    # Code / Math explicit in part
    assert extract_language("some_dataset", "ingredient2-cranecode") == "Code"
    assert extract_language("some_dataset", "finemath-3plus") == "Math"

    # Math / Code explicit in dataset
    assert extract_language("starcoder", "") == "Code"
    assert extract_language("megamath", "") == "Math"
    assert extract_language("swallow-code", "") == "Code"

    # Default to English
    assert extract_language("dclm-1.0", "") == "eng_Latn"
    assert extract_language("common-pile-stackv2-0.1", "documents") == "eng_Latn"


def test_get_source_type():
    assert get_source_type("hplt-4.0") == "Web Data"
    assert get_source_type("dclm-1.0") == "Web Data"
    assert get_source_type("nemotron-cc-1.0") == "Web Data"

    assert get_source_type("starcoder-0.0.0") == "Code & Math"
    assert get_source_type("megamath-0.0.0") == "Code & Math"
    assert get_source_type("common-pile-stackv2-0.1") == "Code & Math"

    assert get_source_type("finewiki-0.0.0") == "Curated (Wiki/PDF/Books)"
    assert get_source_type("finepdfs-1.0.0") == "Curated (Wiki/PDF/Books)"
    assert get_source_type("agenttrove-0.0") == "Curated (Wiki/PDF/Books)"

    assert get_source_type("nemotron-cc-tower+-0.1") == "Translations / Synthetic"
    assert get_source_type("dochplt-3.1") == "Translations / Synthetic"
