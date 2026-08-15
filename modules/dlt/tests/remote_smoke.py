import json
import os

from igor_dlt import IngestConfig, ingest_records


rows = [{"record_id": "ticket-001", "channel": "email", "text": "I cannot reset my password."}]
root = os.environ["IGOR_LANCE_URI"]
support = ingest_records(rows, IngestConfig("qualification", "support", root, "qualification/support", storage_options={"allow_http": "true"}))
finance = ingest_records(rows, IngestConfig("qualification", "finance", root, "qualification/finance", storage_options={"allow_http": "true"}))
assert support.records[0]["domain"] == "support"
assert finance.records[0]["domain"] == "finance"
assert support.config.namespace_name != finance.config.namespace_name
assert "pipeline" in support.load_metadata
print(json.dumps({"support": support.as_dict(), "finance": finance.as_dict()}, sort_keys=True))
