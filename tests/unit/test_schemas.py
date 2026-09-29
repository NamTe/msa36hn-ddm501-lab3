"""Unit tests for request and response validation contracts."""

import pytest
from pydantic import ValidationError

from app.schemas import (
    BatchPredictionRequest,
    HealthResponse,
    PredictionItem,
    PredictionRequest,
    PredictionResponse,
)


class TestPredictionRequest:
    def test_valid_request(self):
        request = PredictionRequest(user_id="196", movie_id="242")
        assert request.user_id == "196"
        assert request.movie_id == "242"

    def test_valid_request_with_numeric_strings(self):
        request = PredictionRequest(user_id="123", movie_id="456")
        assert request.user_id == "123"
        assert request.movie_id == "456"

    def test_missing_user_id_raises_error(self):
        with pytest.raises(ValidationError) as exc:
            PredictionRequest(movie_id="242")
        assert [(e["loc"], e["type"]) for e in exc.value.errors()] == [(("user_id",), "missing")]

    def test_missing_movie_id_raises_error(self):
        with pytest.raises(ValidationError) as exc:
            PredictionRequest(user_id="196")
        assert [(e["loc"], e["type"]) for e in exc.value.errors()] == [(("movie_id",), "missing")]

    def test_missing_both_fields_raises_error(self):
        with pytest.raises(ValidationError) as exc:
            PredictionRequest()
        assert {e["loc"] for e in exc.value.errors()} == {("user_id",), ("movie_id",)}
        assert all(e["type"] == "missing" for e in exc.value.errors())

    def test_empty_user_id_raises_error(self):
        with pytest.raises(ValidationError):
            PredictionRequest(user_id="", movie_id="242")

    @pytest.mark.parametrize("user_id", [" ", "   ", "\t\n"])
    def test_whitespace_only_user_id_raises_error(self, user_id):
        with pytest.raises(ValidationError):
            PredictionRequest(user_id=user_id, movie_id="242")

    @pytest.mark.parametrize("field", ["user_id", "movie_id"])
    def test_none_values_raise_error(self, field):
        payload = {"user_id": "196", "movie_id": "242"}
        payload[field] = None
        with pytest.raises(ValidationError) as exc:
            PredictionRequest(**payload)
        assert exc.value.errors()[0]["loc"] == (field,)
        assert exc.value.errors()[0]["type"] == "string_type"

    def test_integer_user_id_raises_error(self):
        """Pydantic v2 does not coerce integer IDs to strings by default."""
        with pytest.raises(ValidationError) as exc:
            PredictionRequest(user_id=196, movie_id="242")
        assert exc.value.errors()[0]["loc"] == ("user_id",)
        assert exc.value.errors()[0]["type"] == "string_type"

    def test_surrounding_whitespace_is_stripped(self):
        request = PredictionRequest(user_id=" 196 ", movie_id="\t242\n")
        assert request.user_id == "196"
        assert request.movie_id == "242"


class TestPredictionResponse:
    def test_valid_response(self):
        response = PredictionResponse(
            user_id="196", movie_id="242", predicted_rating=3.5, model_version="1.0.0"
        )
        assert response.model_dump() == {
            "user_id": "196",
            "movie_id": "242",
            "predicted_rating": 3.5,
            "model_version": "1.0.0",
        }

    def test_rating_below_minimum_raises_error(self):
        with pytest.raises(ValidationError):
            PredictionResponse(
                user_id="196", movie_id="242", predicted_rating=0.99, model_version="1.0.0"
            )

    def test_rating_above_maximum_raises_error(self):
        with pytest.raises(ValidationError):
            PredictionResponse(
                user_id="196", movie_id="242", predicted_rating=5.01, model_version="1.0.0"
            )

    @pytest.mark.parametrize("rating", [1.0, 5.0])
    def test_rating_at_boundaries(self, rating):
        response = PredictionResponse(
            user_id="196", movie_id="242", predicted_rating=rating, model_version="1.0.0"
        )
        assert response.predicted_rating == rating


class TestHealthResponse:
    def test_valid_health_response(self):
        response = HealthResponse(status="healthy", model_loaded=True)
        assert response.model_dump() == {"status": "healthy", "model_loaded": True}

    @pytest.mark.parametrize("status", ["healthy", "unhealthy", "degraded"])
    @pytest.mark.parametrize("model_loaded", [True, False])
    def test_health_response_status_types(self, status, model_loaded):
        response = HealthResponse(status=status, model_loaded=model_loaded)
        assert response.status == status
        assert response.model_loaded is model_loaded


class TestBatchPredictionRequest:
    def test_valid_batch_request(self, sample_batch_request):
        request = BatchPredictionRequest(**sample_batch_request)
        assert len(request.predictions) == 3
        assert all(isinstance(item, PredictionItem) for item in request.predictions)
        assert [(item.user_id, item.movie_id) for item in request.predictions] == [
            ("196", "242"),
            ("186", "302"),
            ("22", "377"),
        ]

    def test_empty_predictions_list_raises_error(self):
        with pytest.raises(ValidationError):
            BatchPredictionRequest(predictions=[])

    def test_too_many_predictions_raises_error(self):
        with pytest.raises(ValidationError):
            BatchPredictionRequest(predictions=[{"user_id": "196", "movie_id": "242"}] * 101)

    @pytest.mark.parametrize("count", [1, 100])
    def test_batch_size_boundaries(self, count):
        request = BatchPredictionRequest(
            predictions=[{"user_id": "196", "movie_id": "242"}] * count
        )
        assert len(request.predictions) == count
