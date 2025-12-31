from phylogenic.benchmark.utils import check_answer


def test_check_answer_multiple_choice():
    assert check_answer("Answer: C.", "C") is True
    assert check_answer("I think it's B", "B") is True
    assert check_answer("Option (A)", "A") is True


def test_check_answer_numerical_exact():
    assert check_answer("The final answer: 6", "6") is True
    assert check_answer("Result is 60", "60") is True


def test_check_answer_numerical_false_positive():
    # Ensure '16' does not match expected '6'
    assert check_answer("The answer is 16", "6") is False
    assert check_answer("Answer: 216", "16") is False


def test_check_answer_text_substring():
    assert check_answer("Paris is the capital of France", "Paris") is True
    assert check_answer("The chemical symbol is Au", "Au") is True


def test_check_answer_case_insensitive():
    assert check_answer("paris is lovely", "Paris") is True
    assert check_answer("answer: b", "B") is True


def test_check_answer_negative_and_variations():
    # Avoid false positives ("Comparison" should not match "Paris")
    assert check_answer("This is a comparison of methods", "Paris") is False

    # More multiple choice variations
    assert check_answer("Answer: (B)", "B") is True
    assert check_answer("B)", "B") is True
    assert check_answer("option a.", "A") is True
    assert check_answer("final answer: c", "C") is True

    # Numeric edge cases: leading zeros and decimal forms
    assert check_answer("Result: 06", "6") is True
    assert check_answer("Final value: 6.0", "6.0") is True
    assert check_answer("The number is 1,234", "1234") is False  # comma formatting not recognized as numeric token
