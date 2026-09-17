import pytest
from src.analyze import extract_language

def test_extract_language():
    assert extract_language("eng_Latn") == "eng_Latn"
    assert extract_language("bos_Latn-eng_Latn") == "bos_Latn"
    assert extract_language("eng_Latn-als_Latn") == "als_Latn"
    assert extract_language("clean/als_Latn") == "als_Latn"
    assert extract_language("parallel/tower72b/deu_Latn") == "deu_Latn"
    assert extract_language("ingredient2-cranecode") == "Code"
    assert extract_language("finemath-3plus") == "Math"
    assert extract_language("documents") == "Other"
    assert extract_language(None) == "Other"
