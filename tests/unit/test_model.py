"""Unit tests for model loading, prediction, and error handling."""

import pytest

from app.model import MovieRatingModel


class TestMovieRatingModel:
    """Exercise the model wrapper using the shared trained model."""

    def test_model_loads_successfully(self, trained_model):
        assert trained_model is not None
        assert trained_model.is_loaded()

    def test_model_instance_has_model_attribute(self, trained_model):
        assert trained_model.model is not None

    def test_predict_returns_float(self, trained_model):
        result = trained_model.predict("196", "242")
        assert isinstance(result, float)

    def test_predict_returns_value_in_valid_range(self, trained_model):
        result = trained_model.predict("196", "242")
        assert 1.0 <= result <= 5.0

    def test_predict_multiple_pairs_all_in_range(self, trained_model, known_user_movie_pairs):
        for pair in known_user_movie_pairs:
            result = trained_model.predict(pair["user_id"], pair["movie_id"])
            assert 1.0 <= result <= 5.0, pair

    def test_predict_batch_returns_list(self, trained_model):
        results = trained_model.predict_batch([("196", "242"), ("186", "302")])
        assert isinstance(results, list)

    def test_predict_batch_returns_correct_length(self, trained_model):
        pairs = [("196", "242"), ("186", "302"), ("22", "377")]
        results = trained_model.predict_batch(pairs)
        assert len(results) == len(pairs)

    def test_predict_batch_all_values_in_range(self, trained_model):
        pairs = [("196", "242"), ("186", "302"), ("22", "377")]
        results = trained_model.predict_batch(pairs)
        assert len(results) == len(pairs)
        for rating in results:
            assert isinstance(rating, float)
            assert 1.0 <= rating <= 5.0

    def test_is_loaded_returns_bool(self, trained_model):
        assert isinstance(trained_model.is_loaded(), bool)

    def test_is_loaded_returns_true_for_loaded_model(self, trained_model):
        assert trained_model.is_loaded() is True

    def test_predict_with_none_user_id(self, trained_model):
        """Surprise treats None as an unknown user and returns a fallback rating."""
        result = trained_model.predict(None, "242")
        assert isinstance(result, float)
        assert 1.0 <= result <= 5.0

    @pytest.mark.parametrize("pairs", [("", "242"), ("196", ""), ("", "")])
    def test_predict_with_empty_string(self, trained_model, pairs):
        """Unknown IDs are handled by the model; API schemas validate input separately."""
        result = trained_model.predict(*pairs)
        assert isinstance(result, float)
        assert 1.0 <= result <= 5.0

    def test_predict_batch_with_no_pairs(self, trained_model):
        assert trained_model.predict_batch([]) == []


class TestModelFileHandling:
    def test_model_raises_error_for_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            MovieRatingModel(model_path=str(tmp_path / "missing.pkl"))
