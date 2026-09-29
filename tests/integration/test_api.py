"""Integration tests exercising API routing, validation, and real model predictions."""

import pytest


class TestHealthEndpoint:
    def test_health_returns_200(self, test_client):
        assert test_client.get("/health").status_code == 200

    def test_health_response_has_status_field(self, test_client):
        assert test_client.get("/health").json()["status"] == "healthy"

    def test_health_response_has_model_loaded_field(self, test_client):
        assert test_client.get("/health").json()["model_loaded"] is True

    def test_health_model_loaded_is_boolean(self, test_client):
        assert isinstance(test_client.get("/health").json()["model_loaded"], bool)


class TestRootEndpoint:
    def test_root_returns_200(self, test_client):
        assert test_client.get("/").status_code == 200

    def test_root_contains_api_info(self, test_client):
        data = test_client.get("/").json()
        for field in ("name", "version"):
            assert isinstance(data[field], str)
            assert data[field]
        assert data["docs"] == "/docs"
        assert test_client.get(data["docs"]).status_code == 200


class TestPredictEndpoint:
    def test_predict_valid_request_returns_200(self, test_client, sample_prediction_request):
        response = test_client.post("/predict", json=sample_prediction_request)
        assert response.status_code == 200

    def test_predict_response_has_predicted_rating(self, test_client, sample_prediction_request):
        response = test_client.post("/predict", json=sample_prediction_request)
        assert response.status_code == 200
        assert isinstance(response.json()["predicted_rating"], float)

    def test_predict_response_has_user_id(self, test_client, sample_prediction_request):
        response = test_client.post("/predict", json=sample_prediction_request)
        assert response.status_code == 200
        assert response.json()["user_id"] == sample_prediction_request["user_id"]

    def test_predict_response_has_movie_id(self, test_client, sample_prediction_request):
        response = test_client.post("/predict", json=sample_prediction_request)
        assert response.status_code == 200
        assert response.json()["movie_id"] == sample_prediction_request["movie_id"]

    def test_predict_response_rating_in_valid_range(self, test_client, sample_prediction_request):
        response = test_client.post("/predict", json=sample_prediction_request)
        assert response.status_code == 200
        assert 1.0 <= response.json()["predicted_rating"] <= 5.0

    def test_predict_missing_user_id_returns_422(self, test_client):
        response = test_client.post("/predict", json={"movie_id": "242"})
        assert response.status_code == 422
        assert response.json()["detail"][0]["loc"] == ["body", "user_id"]

    def test_predict_missing_movie_id_returns_422(self, test_client):
        response = test_client.post("/predict", json={"user_id": "196"})
        assert response.status_code == 422
        assert response.json()["detail"][0]["loc"] == ["body", "movie_id"]

    def test_predict_empty_body_returns_422(self, test_client):
        assert test_client.post("/predict").status_code == 422
        assert test_client.post("/predict", json={}).status_code == 422

    def test_predict_invalid_json_returns_422(self, test_client):
        response = test_client.post(
            "/predict", content="invalid json", headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422
        assert response.json()["detail"][0]["type"] == "json_invalid"

    def test_predict_multiple_valid_requests(self, test_client, known_user_movie_pairs):
        for pair in known_user_movie_pairs:
            payload = {"user_id": pair["user_id"], "movie_id": pair["movie_id"]}
            response = test_client.post("/predict", json=payload)
            assert response.status_code == 200, payload
            data = response.json()
            assert data["user_id"] == payload["user_id"]
            assert data["movie_id"] == payload["movie_id"]
            assert 1.0 <= data["predicted_rating"] <= 5.0

    @pytest.mark.parametrize("field", ["user_id", "movie_id"])
    @pytest.mark.parametrize("value", ["", "   ", None, 196, "1" * 10000])
    def test_predict_rejects_invalid_ids(self, test_client, field, value):
        payload = {"user_id": "196", "movie_id": "242", field: value}
        response = test_client.post("/predict", json=payload)
        assert response.status_code == 422
        assert response.json()["detail"][0]["loc"] == ["body", field]


class TestBatchPredictEndpoint:
    def test_batch_predict_returns_200(self, test_client, sample_batch_request):
        response = test_client.post("/predict/batch", json=sample_batch_request)
        assert response.status_code == 200

    def test_batch_predict_returns_correct_count(self, test_client, sample_batch_request):
        response = test_client.post("/predict/batch", json=sample_batch_request)
        assert response.status_code == 200
        data = response.json()
        assert data["total_count"] == len(sample_batch_request["predictions"])
        assert len(data["predictions"]) == data["total_count"]
        assert [
            {"user_id": item["user_id"], "movie_id": item["movie_id"]}
            for item in data["predictions"]
        ] == sample_batch_request["predictions"]

    def test_batch_predict_all_ratings_in_range(self, test_client, sample_batch_request):
        response = test_client.post("/predict/batch", json=sample_batch_request)
        assert response.status_code == 200
        predictions = response.json()["predictions"]
        assert len(predictions) == len(sample_batch_request["predictions"])
        for prediction in predictions:
            assert isinstance(prediction["predicted_rating"], float)
            assert 1.0 <= prediction["predicted_rating"] <= 5.0

    @pytest.mark.parametrize("count", [0, 101])
    def test_invalid_batch_size_returns_422(self, test_client, count):
        response = test_client.post(
            "/predict/batch",
            json={"predictions": [{"user_id": "196", "movie_id": "242"}] * count},
        )
        assert response.status_code == 422


class TestErrorHandling:
    def test_404_for_unknown_endpoint(self, test_client):
        assert test_client.get("/unknown").status_code == 404

    def test_method_not_allowed_get_predict(self, test_client):
        assert test_client.get("/predict").status_code == 405

    def test_method_not_allowed_post_health(self, test_client):
        assert test_client.post("/health").status_code == 405


class TestModelInfoEndpoint:
    def test_model_info_returns_200(self, test_client):
        assert test_client.get("/model/info").status_code == 200

    def test_model_info_has_version(self, test_client, sample_prediction_request):
        data = test_client.get("/model/info").json()
        assert isinstance(data["model_version"], str)
        assert data["model_version"]
        prediction = test_client.post("/predict", json=sample_prediction_request)
        assert prediction.status_code == 200
        assert prediction.json()["model_version"] == data["model_version"]

    def test_model_info_has_is_loaded(self, test_client):
        assert test_client.get("/model/info").json()["is_loaded"] is True
