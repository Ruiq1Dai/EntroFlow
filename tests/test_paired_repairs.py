from experiments.musique.evaluate_paired_repairs import compare


def test_compare_reports_rescue_regression_and_cost():
    baseline = {
        "2hop-a": {"em": False, "total_tokens": 10},
        "3hop-b": {"em": True, "total_tokens": 20},
    }
    candidate = {
        "2hop-a": {"em": True, "total_tokens": 14},
        "3hop-b": {"em": False, "total_tokens": 24},
    }
    report = compare(baseline, candidate)
    assert report["rescued"] == 1
    assert report["regressed"] == 1
    assert report["net_gain"] == 0
    assert report["mean_token_delta"] == 4
