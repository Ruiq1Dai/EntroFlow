from entroflow.evaluation import exact_match_any, token_f1_any


def test_exact_match_accepts_alias_but_not_explanation():
    answers = ("Randall County", "Randall County, Texas")
    assert exact_match_any("Randall County, Texas", answers)
    assert not exact_match_any("The stadium is in Randall County, Texas.", answers)


def test_token_f1_uses_best_alias():
    answers = ("Anderson Tapes", "The Anderson Tapes")
    assert token_f1_any("The Anderson Tapes", answers) == 1.0
