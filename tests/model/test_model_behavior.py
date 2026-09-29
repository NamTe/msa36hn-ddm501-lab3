"""Behavioral checks against the trained movie-rating model.

Accuracy thresholds apply to the lab's known examples, not a held-out evaluation set.
"""

import pytest


class TestModelInvariance:
    def test_same_input_same_output(self, trained_model):
        first = trained_model.predict("196", "242")
        assert trained_model.predict("196", "242") == first

    def test_multiple_calls_consistent(self, trained_model):
        results = [trained_model.predict("196", "242") for _ in range(5)]
        assert all(result == results[0] for result in results)

    def test_batch_order_independent(self, trained_model):
        """Preserve pair associations and duplicates when reordering a batch."""
        pairs = [("196", "242"), ("186", "302"), ("22", "377"), ("196", "242")]
        permutation = [2, 0, 3, 1]
        original = trained_model.predict_batch(pairs)
        reordered = trained_model.predict_batch([pairs[index] for index in permutation])
        assert len(original) == len(pairs)
        assert len(reordered) == len(pairs)
        assert reordered == [original[index] for index in permutation]

    def test_individual_vs_batch_same_results(self, trained_model):
        pairs = [("196", "242"), ("186", "302"), ("22", "377")]
        individual = [trained_model.predict(user, movie) for user, movie in pairs]
        assert trained_model.predict_batch(pairs) == individual


class TestModelDirectional:
    """Check the lab's accuracy and input-sensitivity expectations."""

    def test_predictions_are_reasonable(self, trained_model, known_user_movie_pairs):
        assert known_user_movie_pairs
        for pair in known_user_movie_pairs:
            prediction = trained_model.predict(pair["user_id"], pair["movie_id"])
            assert (
                abs(prediction - pair["actual_rating"]) < 1.5
            ), f"{pair}: predicted {prediction}, error must be below 1.5"

    def test_different_movies_different_predictions(self, trained_model):
        predictions = [
            trained_model.predict("196", movie_id)
            for movie_id in ("242", "302", "377", "51", "346")
        ]
        assert len(set(predictions)) > 1, "Changing movies never changes the prediction"

    def test_different_users_different_predictions(self, trained_model):
        predictions = [
            trained_model.predict(user_id, "242") for user_id in ("196", "186", "22", "244", "166")
        ]
        assert len(set(predictions)) > 1, "Changing users never changes the prediction"


class TestMinimumFunctionality:
    def test_can_predict_for_known_user(self, trained_model):
        prediction = trained_model.predict("196", "242")
        assert isinstance(prediction, float)
        assert 1.0 <= prediction <= 5.0

    def test_can_predict_for_multiple_users(self, trained_model, known_user_movie_pairs):
        assert len({pair["user_id"] for pair in known_user_movie_pairs}) > 1
        for pair in known_user_movie_pairs:
            prediction = trained_model.predict(pair["user_id"], pair["movie_id"])
            assert isinstance(prediction, float)
            assert 1.0 <= prediction <= 5.0, pair

    def test_predictions_not_all_same(self, trained_model, known_user_movie_pairs):
        predictions = [
            trained_model.predict(pair["user_id"], pair["movie_id"])
            for pair in known_user_movie_pairs
        ]
        assert len(set(predictions)) > 1, "All predictions are identical"

    def test_handles_unknown_user_gracefully(self, trained_model, unknown_users):
        """The current SVD model returns fallback ratings for unknown users."""
        assert unknown_users
        for user_id in unknown_users:
            prediction = trained_model.predict(user_id, "242")
            assert isinstance(prediction, float)
            assert 1.0 <= prediction <= 5.0, user_id

    def test_handles_unknown_movie_gracefully(self, trained_model, unknown_movies):
        assert unknown_movies
        for movie_id in unknown_movies:
            prediction = trained_model.predict("196", movie_id)
            assert isinstance(prediction, float)
            assert 1.0 <= prediction <= 5.0, movie_id


class TestModelPerformance:
    def test_average_error_acceptable(self, trained_model, known_user_movie_pairs):
        assert known_user_movie_pairs
        errors = [
            abs(trained_model.predict(pair["user_id"], pair["movie_id"]) - pair["actual_rating"])
            for pair in known_user_movie_pairs
        ]
        mean_absolute_error = sum(errors) / len(errors)
        assert mean_absolute_error < 1.0, f"MAE {mean_absolute_error:.3f} must be below 1.0"

    def test_no_extreme_errors(self, trained_model, known_user_movie_pairs):
        assert known_user_movie_pairs
        for pair in known_user_movie_pairs:
            prediction = trained_model.predict(pair["user_id"], pair["movie_id"])
            assert (
                abs(prediction - pair["actual_rating"]) <= 3.0
            ), f"{pair}: predicted {prediction}, error must not exceed 3.0"


class TestModelRobustness:
    @pytest.mark.parametrize("user_id,movie_id", [("196", "242"), ("1", "1"), ("22", "377")])
    def test_handles_string_numeric_ids(self, trained_model, user_id, movie_id):
        prediction = trained_model.predict(user_id, movie_id)
        assert isinstance(prediction, float)
        assert 1.0 <= prediction <= 5.0

    @pytest.mark.parametrize("user_id,movie_id", [("001", "1"), ("1", "001"), ("001", "001")])
    def test_handles_leading_zeros_in_ids(self, trained_model, user_id, movie_id):
        """Padded IDs may be distinct; require valid, repeatable fallback behavior."""
        prediction = trained_model.predict(user_id, movie_id)
        assert isinstance(prediction, float)
        assert 1.0 <= prediction <= 5.0
        assert trained_model.predict(user_id, movie_id) == prediction
