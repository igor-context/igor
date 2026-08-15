"""Qualification smoke test for an S3-compatible LanceDB endpoint."""

import os
import time
import urllib.request

from igor_lancedb import LanceStore, LanceStoreConfig


ENDPOINT = os.environ.get("IGOR_S3_ENDPOINT", "http://minio:9000")
ROOT_URI = os.environ.get("IGOR_LANCE_URI", "s3://igor-lance/qualification")


def wait_for_minio() -> None:
    health_url = f"{ENDPOINT}/minio/health/live"
    for _ in range(30):
        try:
            with urllib.request.urlopen(health_url, timeout=2) as response:
                if response.status == 200:
                    return
        except OSError:
            time.sleep(1)
    raise RuntimeError(f"MinIO did not become healthy at {health_url}")


def store(domain: str) -> LanceStore:
    return LanceStore(
        LanceStoreConfig(
            ROOT_URI,
            "qualification",
            domain,
            {
                "aws_access_key_id": os.environ["AWS_ACCESS_KEY_ID"],
                "aws_secret_access_key": os.environ["AWS_SECRET_ACCESS_KEY"],
                "aws_region": os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
                "aws_endpoint_url": ENDPOINT,
                "allow_http": "true",
                "aws_virtual_hosted_style_request": "false",
            },
        )
    )


def main() -> None:
    wait_for_minio()
    support = store("support")
    if "events" in support.names():
        support.replace("events", [{"id": 1, "amount": 2}])
    else:
        support.create("events", [{"id": 1, "amount": 2}])
    support.add("events", [{"id": 2, "amount": 3}])
    expected = [
        {"id": 1, "amount": 2},
        {"id": 2, "amount": 3},
    ]
    assert support.read("events").to_pylist() == expected

    reopened = store("support")
    assert reopened.read("events").to_pylist() == expected
    assert reopened.names() == ["events"]

    finance = store("finance")
    assert finance.names() == []
    try:
        finance.read("events")
    except ValueError as error:
        assert str(error) == "unknown Lance table: events"
    else:
        raise AssertionError("finance namespace discovered support table")
    print("remote LanceStore smoke passed: namespaced create, add, reopen, list, and isolation")


if __name__ == "__main__":
    main()
