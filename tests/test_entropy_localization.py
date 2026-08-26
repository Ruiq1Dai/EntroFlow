from experiments.musique.evaluate_entropy_localization import localization_accuracy


def test_localization_accuracy_uses_independent_audit_labels():
    diagnostics = [
        {
            "example_id": "x",
            "edge_scores": [
                {"role": "planner", "entropy": 0.1},
                {"role": "evidence", "entropy": 0.8},
            ],
        }
    ]
    audit = [{"example_id": "x", "first_error_role": "evidence"}]
    result = localization_accuracy(diagnostics, audit, "entropy")
    assert result["hits"] == 1
    assert result["accuracy"] == 1.0
