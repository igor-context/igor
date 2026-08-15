# Hugging Face document qualification v0.1

Four-page/eight-question deterministic qualification pack for the pinned
`lance-format/docvqa-lance` MIT dataset. The checked-in fixture is a reviewed,
redacted projection used to qualify the seams without downloading supplied answers,
embeddings, or indices. A live acquisition must replace only the page fixture after
verifying the pinned Hub revision and terms.

The pack selects the reusable `document.qa.v1` Context Model and prescribes no OCR
adapter or provider implementation.

`domain-schema.yaml` mirrors the exact columns of the pinned Hugging Face revision and
documents how each column may be used. Human reference answers and supplied embeddings
are explicitly evaluation/retrieval-only and cannot enter a completion prompt.
`semantic-definition.yaml` describes the source `question_types` taxonomy and the
general answerability rules. The runner projects only the tags relevant to each
question into the multimodal request. Both YAML documents participate in work identity,
so an approved semantic change invalidates stale cached enrichment.
