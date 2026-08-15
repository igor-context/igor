from igor_context_compiler import CompilationRequest, ContextModelCompileRequest, DependencyInventory, DerivationSpec, plan


def test_public_models_reject_unknown_fields():
    try:
        CompilationRequest(run_identity="x", task_id="t", required_output_identities=("x",),
                           code_revision="c", unexpected=True)
    except Exception as error:
        assert "unexpected" in str(error)
    else:
        raise AssertionError("unknown fields must be rejected")


def test_context_model_request_rejects_unknown_fields():
    try:
        ContextModelCompileRequest(declaration={"context_model": {"id": "x", "revision": "1"}}, unexpected=True)
    except Exception as error:
        assert "unexpected" in str(error)
    else:
        raise AssertionError("unknown fields must be rejected")
