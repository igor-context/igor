# Hugging Face module

Owns public Hugging Face repository discovery and bounded raw acquisition. The
adapter is deliberately independent of support meaning, dlt, the compiler, and
the evaluator.

## Interface

```text
DatasetSelection(repository, revision, configuration, split, file, row_numbers,
                 expected_file_sha256, expected_row_sha256, fields, license)
HuggingFaceAdapter.acquire(selection) -> AcquisitionResult
HuggingFaceAdapter.acquire_document(selection, output_dir) -> DocumentAcquisitionResult
HuggingFaceAdapter.acquire_images(selection, output_dir) -> ImageAcquisitionResult
RowSelection(repository, revision, config, split, selector_field,
             selector_values, fields, license, row_numbers, max_bytes, max_rows)
HuggingFaceAdapter.acquire_rows(selection) -> RowAcquisitionResult
```

`revision` must be a 40-character commit SHA. Acquisition verifies repository
metadata, declared license, exact file bytes, selected row identities, stable
ordering, and per-row content hashes. The returned records contain only the
declared `fields`; labels or judgments are therefore excluded by the caller's
selection manifest rather than hidden inside the adapter. Credentials are read
only from `HF_TOKEN` at runtime and are never serialized.

Errors fail closed for mutable revisions, missing files/rows, schema drift,
duplicate row identities, hash mismatch, gated/private repositories, and
disallowed licenses.

`RowSelection` is the Dataset Viewer seam for bounded field/value acquisition. The
selector field and values are data supplied by the scenario; the adapter contains no
dataset, legal-domain, taxonomy, or amendment field names. Optional explicit row
offsets keep sparse selections bounded without scanning unrelated rows.

## Commands

- `docker compose run --rm --build huggingface-test`
- `docker compose --profile qualification run --rm --build huggingface-qualification`

The qualification command writes generated metadata under `.igor/` and never
commits downloaded source data.

The document command is:

```text
docker compose --profile check run --rm --build huggingface-document-acquisition
```

It writes `acquisition.json`, a sealed evaluator-only `judgments.json`, and
content-addressed selected page images under `.igor/huggingface-document-acquisition/`.
Supplied answers and embeddings are excluded from acquisition records.

The image command is:

```text
docker compose --profile qualification run --rm --build huggingface-image-qualification
```

It verifies the pinned bounded ZIP archive, extracts selected image bytes with
content hashes, and writes evaluator-only labels to a separate judgment sidecar.
