import pytest
from fastapi.testclient import TestClient

from app.main import app

DECAN = dict(
    municipality="Deqan", elevation=678, distance_from_river=0.79, rainfall=800,
    soil_type="Aluviale", max_water_level=97, min_water_level=27, min_slope=3, max_slope=70,
)


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:  
        yield test_client


def test_health(client):
    assert client.get("/health").json() == {"status": "healthy"}


def test_predict_known_municipality(client):
    response = client.post("/api/flood-risk/predict", json=DECAN)
    assert response.status_code == 200
    body = response.json()
    assert body["risk"] == "Medium" and body["risk_code"] == 1
    assert body["municipality"] == "Deqan"
    assert set(body["class_scores"]) == {"Low", "Medium", "High"}
    assert sum(body["class_scores"].values()) == pytest.approx(1.0, abs=0.01)
    assert body["warnings"] == []
    assert body["model"]["name"] == "random_forest"


def test_municipality_is_optional(client):
    payload = {k: v for k, v in DECAN.items() if k != "municipality"}
    assert client.post("/api/flood-risk/predict", json=payload).status_code == 200


def test_out_of_range_input_still_predicts_but_warns(client):
    body = client.post("/api/flood-risk/predict", json=DECAN | {"distance_from_river": 50}).json()
    assert any("distance_from_river" in w for w in body["warnings"])
    assert body["out_of_range"] == [
        {"feature": "distance_from_river", "value": 50.0, "training_min": 0.05, "training_max": 14.16, "unit": "km"}
    ]


def test_cors_allows_vite_dev_server(client):
    response = client.options(
        "/api/flood-risk/predict",
        headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST"},
    )
    assert response.headers.get("access-control-allow-origin") == "http://localhost:5173"


@pytest.mark.parametrize(
    "payload",
    [
        DECAN | {"soil_type": "Sand"},                               # unknown category
        {k: v for k, v in DECAN.items() if k != "rainfall"},          # missing field
        DECAN | {"rainfall": -10},                                   # negative value
        DECAN | {"min_water_level": 200, "max_water_level": 100},    # min > max
        DECAN | {"rainfall": "a lot"},                               # wrong type
        DECAN | {"risk": 2},                                         # unexpected field
    ],
    ids=["unknown_soil", "missing_field", "negative", "min_gt_max", "wrong_type", "extra_field"],
)
def test_invalid_input_returns_422(client, payload):
    assert client.post("/api/flood-risk/predict", json=payload).status_code == 422


def test_metadata(client):
    body = client.get("/api/flood-risk/metadata").json()
    assert len(body["soil_types"]) == 7
    assert {f["name"] for f in body["numeric_features"]} == {
        "elevation", "distance_from_river", "rainfall", "max_water_level",
        "min_water_level", "min_slope", "max_slope",
    }
    assert body["risk_levels"] == {"0": "Low", "1": "Medium", "2": "High"}
