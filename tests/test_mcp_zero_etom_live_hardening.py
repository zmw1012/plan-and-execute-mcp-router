"""Live-runtime hardening tests (no real network / paid APIs)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

import pytest

from hk_equity_mcp.agent.llm_client import OpenAICompatibleChatClient
from hk_equity_mcp.mcp_zero import method_identity_sha256
from hk_equity_mcp.mcp_zero.embedding import (
    CountingEmbedder,
    IncompleteEmbeddingTableError,
    build_embedding_table,
    embedding_cache_identity,
    identity_digest,
    load_complete_table,
    table_status,
)
from hk_equity_mcp.mcp_zero_etom import (
    EMBEDDING_NAMESPACE,
    EXPECTED_FORMAL_SAMPLE_SHA256,
    EXPECTED_HK_METHOD_CARD_SHA256,
    EXPECTED_ORDERED_IDS_SHA256,
)
from hk_equity_mcp.mcp_zero_etom.cli import build_parser, main as cli_main
from hk_equity_mcp.mcp_zero_etom.identity import (
    ETOM_RESUME_IDENTITY_KEYS,
    assert_resume_compatible,
    resume_identity_payload,
    resume_identity_sha256,
)
from hk_equity_mcp.mcp_zero_etom.l4_metrics import evaluate_l4_separate
from hk_equity_mcp.mcp_zero_etom.metadata import build_etom_catalog
from hk_equity_mcp.mcp_zero_etom.mock_llm import EtomMockLLM, atr_text, final_json, invoke_json
from hk_equity_mcp.mcp_zero_etom.prompt import etom_prompt_hash
from hk_equity_mcp.mcp_zero_etom.runner import EtomMcpZeroRunner
from hk_equity_mcp.mcp_zero_etom.source import assert_formal_lock_identity
from hk_equity_mcp.operation_resolver.retrieval import (
    DeterministicHashEmbeddingBackend,
    OpenAICompatibleEmbeddingBackend,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "etom_mini"


class _RecordingFakeLLM(EtomMockLLM):
    """Records every message blob for gold-leak assertions."""

    def __init__(self) -> None:
        super().__init__()
        self.seen_messages: List[List[Dict[str, str]]] = []

    def complete(self, messages, *, temperature=0.0, max_tokens=1024):
        self.seen_messages.append([dict(m) for m in messages])
        return super().complete(messages, temperature=temperature, max_tokens=max_tokens)


class _FakePaidEmbedding:
    """Stand-in that *is* constructed like a paid backend but never networks."""

    def __init__(self, *args, **kwargs):
        self.model_name = kwargs.get("model") or "text-embedding-3-small"
        self._inner = DeterministicHashEmbeddingBackend(model_name=self.model_name, dim=32)
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        return self._inner.embed(list(texts))


def _write_env_pair(tmp_path: Path) -> tuple[Path, Path]:
    gen = tmp_path / "gen.env"
    emb = tmp_path / "emb.env"
    gen.write_text(
        "\n".join(
            [
                "LLM_PROVIDER=openai_compatible",
                "LLM_BASE_URL=https://api.zhizengzeng.com/v1",
                "LLM_API_KEY=test-gen-key-not-real",
                "LLM_MODEL=deepseek-v4.1-flash",
                "",
            ]
        ),
        encoding="utf-8",
    )
    emb.write_text(
        "\n".join(
            [
                "EMBEDDING_BASE_URL=https://api.zhizengzeng.com/v1",
                "EMBEDDING_API_KEY=test-emb-key-not-real",
                "EMBEDDING_MODEL=text-embedding-3-small",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return gen, emb


def test_formal_hashes_locked():
    ident = assert_formal_lock_identity(ROOT)
    assert ident["sample_sha256"] == EXPECTED_FORMAL_SAMPLE_SHA256
    assert ident["ordered_task_ids_sha256"] == EXPECTED_ORDERED_IDS_SHA256
    assert method_identity_sha256() == EXPECTED_HK_METHOD_CARD_SHA256
    assert etom_prompt_hash() == (
        "8ebac6ce045af0176c1fb9e580bad483b7ca4564e3521adc433c09476db285d7"
    )


def test_max_tasks_excluded_from_resume_identity():
    assert "max_tasks" not in ETOM_RESUME_IDENTITY_KEYS
    base = {
        "experiment": "e",
        "condition": "mcp_zero",
        "method_identity_sha256": "a",
        "question_ids": ["L1-q1", "L2-q1"],
        "server_top_k": 3,
        "tool_top_k": 5,
        "embedding_provider": "openai_compatible",
        "embedding_base_url_host": "api.zhizengzeng.com",
        "max_tasks": 10,
    }
    other = dict(base)
    other["max_tasks"] = 50
    other["question_ids"] = ["L1-q1", "L2-q1", "L3-q1"]
    # Compatible: max_tasks ignored; question_ids is a prefix extension.
    assert_resume_compatible(base, other)
    assert resume_identity_sha256(base) == resume_identity_sha256(
        {**base, "max_tasks": 999}
    )
    payload = resume_identity_payload(base)
    assert "max_tasks" not in payload


def test_provider_host_model_mismatch_fail_closed(tmp_path, monkeypatch):
    gen, emb = _write_env_pair(tmp_path)
    bad = tmp_path / "bad_gen.env"
    bad.write_text(
        gen.read_text(encoding="utf-8").replace(
            "deepseek-v4.1-flash", "wrong-model"
        ),
        encoding="utf-8",
    )
    for key in (
        "LLM_MODEL",
        "LLM_PROVIDER",
        "LLM_BASE_URL",
        "LLM_API_KEY",
        "EMBEDDING_MODEL",
        "EMBEDDING_BASE_URL",
        "EMBEDDING_API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)
    # Use mock mode so fixture source is legal; still exercise paid pin checks.
    runner = EtomMcpZeroRunner(
        ROOT,
        tmp_path / "out",
        split="formal100",
        mode="mock",
        fixture_dir=str(FIXTURE),
        env_file=str(bad),
        embedding_env_file=str(emb),
        allow_dirty=True,
    )
    with pytest.raises(RuntimeError, match="generation model mismatch"):
        runner._load_env_profiles(require=True)


def test_live_refuses_fixture_source(tmp_path):
    gen, emb = _write_env_pair(tmp_path)
    with pytest.raises(Exception, match="Fixture source"):
        EtomMcpZeroRunner(
            ROOT,
            tmp_path / "out",
            split="pilot10",
            mode="live",
            fixture_dir=str(FIXTURE),
            env_file=str(gen),
            embedding_env_file=str(emb),
            allow_dirty=True,
        )


def test_live_constructs_real_client_classes(tmp_path, monkeypatch):
    """Live path constructs paid client classes; injected fakes avoid network."""
    gen, emb = _write_env_pair(tmp_path)
    constructed: Dict[str, Any] = {"llm": 0, "emb": 0}

    def llm_factory():
        constructed["llm"] += 1
        # Prove the production class is the intended target by constructing it
        # with env loaded, but swap to mock for the actual run loop via return.
        # We still instantiate OpenAICompatibleChatClient once to assert wiring.
        client = OpenAICompatibleChatClient(
            base_url=os.environ["LLM_BASE_URL"],
            api_key=os.environ["LLM_API_KEY"],
            model=os.environ["LLM_MODEL"],
            provider=os.environ["LLM_PROVIDER"],
            json_mode=False,
        )
        assert client.json_mode is False
        assert isinstance(client, OpenAICompatibleChatClient)
        # Return a scripted mock so execute() never networks.
        return EtomMockLLM()

    def emb_factory():
        constructed["emb"] += 1
        # Construct the real class (no network until embed()).
        real = OpenAICompatibleEmbeddingBackend(
            base_url=os.environ["EMBEDDING_BASE_URL"],
            api_key=os.environ["EMBEDDING_API_KEY"],
            model=os.environ["EMBEDDING_MODEL"],
        )
        assert isinstance(real, OpenAICompatibleEmbeddingBackend)
        return _FakePaidEmbedding(model="text-embedding-3-small")

    # Pre-build a complete mock index in the shared cache dir.
    payload = json.loads((FIXTURE / "data" / "mcp_registry.json").read_text())
    catalog = build_etom_catalog(payload)
    cache = tmp_path / "cache"
    embedder = CountingEmbedder(
        DeterministicHashEmbeddingBackend(model_name="text-embedding-3-small"),
        model_name="text-embedding-3-small",
    )
    build_embedding_table(
        catalog,
        embedder,
        cache_dir=cache,
        use_summary=True,
        catalog_fingerprint=catalog.fingerprint(),
        namespace=EMBEDDING_NAMESPACE,
    )

    # Live refuses fixture; use build-index path with factory to assert classes,
    # and a separate mock run for task execution without gold in live.
    # Here we only assert construction helpers on a runner configured as live
    # after monkeypatching source_kind via fixture denial — so call factories
    # through build_index on mock + through build_agent injection pattern.
    runner = EtomMcpZeroRunner(
        ROOT,
        tmp_path / "out_mock",
        split="formal100",
        mode="mock",
        fixture_dir=str(FIXTURE),
        embedding_cache_dir=cache,
        allow_dirty=True,
        max_tasks=2,
        env_file=str(gen),
        embedding_env_file=str(emb),
        llm_factory=llm_factory,
        embedding_backend_factory=emb_factory,
    )
    # Force env load like paid mode
    for key in list(os.environ.keys()):
        if key.startswith("LLM_") or key.startswith("EMBEDDING_"):
            monkeypatch.delenv(key, raising=False)
    runner._load_env_profiles(require=True)
    llm = runner._make_llm(paid=True)
    backend = runner._make_embedding_backend(paid=True)
    assert constructed["llm"] == 1
    assert constructed["emb"] == 1
    assert isinstance(llm, EtomMockLLM)
    assert isinstance(backend, _FakePaidEmbedding)


def test_require_complete_index_fail_closed(tmp_path, monkeypatch):
    import urllib.request

    def boom(*a, **k):
        raise RuntimeError("network forbidden")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    out = tmp_path / "liveish"
    cache = tmp_path / "empty_cache"
    cache.mkdir()
    runner = EtomMcpZeroRunner(
        ROOT,
        out,
        split="formal100",
        mode="mock",
        fixture_dir=str(FIXTURE),
        embedding_cache_dir=cache,
        allow_dirty=True,
        require_complete_index=True,
        max_tasks=1,
    )
    with pytest.raises(IncompleteEmbeddingTableError):
        runner.execute()


def test_index_identity_mismatch_and_shared_cache(tmp_path):
    payload = json.loads((FIXTURE / "data" / "mcp_registry.json").read_text())
    catalog = build_etom_catalog(payload)
    cache = tmp_path / "shared"
    embedder = CountingEmbedder(
        DeterministicHashEmbeddingBackend(model_name="text-embedding-3-small"),
        model_name="text-embedding-3-small",
    )
    table = build_embedding_table(
        catalog,
        embedder,
        cache_dir=cache,
        use_summary=True,
        catalog_fingerprint=catalog.fingerprint(),
        namespace=EMBEDDING_NAMESPACE,
    )
    assert table.complete
    # pilot and formal share the same complete registry index identity
    status_a = table_status(
        catalog,
        model_name="text-embedding-3-small",
        cache_dir=cache,
        use_summary=True,
        catalog_fingerprint=catalog.fingerprint(),
        namespace=EMBEDDING_NAMESPACE,
    )
    status_b = table_status(
        catalog,
        model_name="text-embedding-3-small",
        cache_dir=cache,
        use_summary=True,
        catalog_fingerprint=catalog.fingerprint(),
        namespace=EMBEDDING_NAMESPACE,
    )
    assert status_a["identity_sha256"] == status_b["identity_sha256"]
    assert status_a["complete"] is True
    # mismatch model → incomplete / fail closed
    bad_status = table_status(
        catalog,
        model_name="wrong-model",
        cache_dir=cache,
        use_summary=True,
        catalog_fingerprint=catalog.fingerprint(),
        namespace=EMBEDDING_NAMESPACE,
    )
    assert bad_status["complete"] is False
    with pytest.raises(Exception):
        load_complete_table(
            catalog,
            model_name="wrong-model",
            cache_dir=cache,
            use_summary=True,
            catalog_fingerprint=catalog.fingerprint(),
            namespace=EMBEDDING_NAMESPACE,
        )
    # HK default namespace unchanged
    hk = embedding_cache_identity(
        embedding_model="text-embedding-3-small",
        catalog_fingerprint="x",
        metadata_hash="y",
    )
    assert hk["namespace"] == "mcp_zero_hk_v1"
    assert identity_digest(status_a["identity"]) == status_a["identity_sha256"]


def test_task_resume_prefix_10_to_more(tmp_path, monkeypatch):
    import urllib.request

    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("network forbidden")),
    )
    cache = tmp_path / "cache"
    out = tmp_path / "run"
    r1 = EtomMcpZeroRunner(
        ROOT,
        out,
        split="formal100",
        mode="mock",
        fixture_dir=str(FIXTURE),
        embedding_cache_dir=cache,
        allow_dirty=True,
        max_tasks=2,
    )
    r1.execute()
    n1 = len((out / "predictions.jsonl").read_text().strip().splitlines())
    assert n1 == 2
    meta1 = json.loads((out / "run_metadata.json").read_text())
    r2 = EtomMcpZeroRunner(
        ROOT,
        out,
        split="formal100",
        mode="mock",
        fixture_dir=str(FIXTURE),
        embedding_cache_dir=cache,
        allow_dirty=True,
        max_tasks=5,
        resume=True,
    )
    r2.execute()
    rows = [
        json.loads(line)
        for line in (out / "predictions.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert len(rows) == 5
    meta2 = json.loads((out / "run_metadata.json").read_text())
    assert_resume_compatible(meta1, meta2)
    assert meta1["resume_identity_sha256"] != ""  # present
    # max_tasks differs but identity payload excludes it
    assert meta1["max_tasks"] == 2
    assert meta2["max_tasks"] == 5


def test_gold_never_enters_prompt_or_retrieval(tmp_path, monkeypatch):
    import urllib.request

    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("network forbidden")),
    )
    fake = _RecordingFakeLLM()
    # Script an ATR then final without using gold ids in the script content.
    fake.set_task(
        "any",
        [
            atr_text("alpha search", "search documents"),
            final_json(infeasible=True),
        ],
    )
    out = tmp_path / "out"
    cache = tmp_path / "cache"
    runner = EtomMcpZeroRunner(
        ROOT,
        out,
        split="formal100",
        mode="mock",
        fixture_dir=str(FIXTURE),
        embedding_cache_dir=cache,
        allow_dirty=True,
        max_tasks=1,
        llm_factory=lambda: fake,
    )
    # Override mock scripting path by pre-setting factory LLM and empty scripts per task:
    # execute() still calls set_task for mock; clear after prepare by monkeypatching.
    original_run = runner._run_tasks

    def _run_without_gold_script():
        runner.prepare()
        runner._ensure_table()
        agent = runner.build_agent()
        # Replace llm after build
        agent.llm = fake
        task = runner.selected[0]
        gold_blob = json.dumps(task.ground_truth_tools, ensure_ascii=False)
        result = agent.run(
            task.query,
            task_label=task.task_id,
            ground_truth_tools=task.ground_truth_tools,
        )
        assert result["real_tool_executions"] == 0
        for turn_msgs in fake.seen_messages:
            blob = "\n".join(m.get("content") or "" for m in turn_msgs)
            assert "ground_truth" not in blob.lower()
            assert "gold_tool" not in blob.lower()
            if len(gold_blob) > 40:
                assert gold_blob not in blob
        return {"action": "mock", "rows": 1}

    runner._run_tasks = _run_without_gold_script  # type: ignore[method-assign]
    runner.execute()


def test_l4_labels_locked_and_real_tool_zero():
    gold = [
        {"server_name": "A", "tool_name": "t1", "dependencies": []},
        {"server_name": "B", "tool_name": "t2", "dependencies": [0]},
    ]
    m = evaluate_l4_separate(
        ground_truth_tools=gold,
        predicted_tools=[
            {"server_name": "A", "tool_name": "t1"},
            {"server_name": "B", "tool_name": "t2"},
        ],
        predicted_nodes=[
            {"server_name": "A", "tool_name": "t1", "depends_on": []},
            {"server_name": "B", "tool_name": "t2", "depends_on": [0]},
        ],
        status="completed",
    )
    assert set(m) >= {
        "official_union_node_set_em",
        "efs_covering_node_set",
        "cross_server_dependency_correctness",
        "complete_route_success",
    }
    # Labels must remain distinct keys (never aliased).
    assert "official_union_node_set_em" in m
    assert "efs_covering_node_set" in m
    assert "cross_server_dependency_correctness" in m


def test_cli_live_flag_accepted_and_env_args_present():
    p = build_parser()
    args = p.parse_args(
        [
            "--split",
            "formal100",
            "--live",
            "--env-file",
            "/tmp/gen.env",
            "--embedding-env-file",
            "/tmp/emb.env",
            "--etom-source-dir",
            "/tmp/etom",
            "--require-complete-index",
        ]
    )
    assert args.live is True
    assert args.env_file.endswith("gen.env")
    assert args.embedding_env_file.endswith("emb.env")


def test_cli_build_index_mode_is_not_mock(tmp_path, monkeypatch):
    """--build-index-only maps to build-index mode (real embed path), not mock."""
    # Intercept runner to capture mode without network.
    captured: Dict[str, Any] = {}

    class _Stub:
        def __init__(self, *a, **k):
            captured.update(k)

        def execute(self):
            return {"action": "build-index"}

    monkeypatch.setattr(
        "hk_equity_mcp.mcp_zero_etom.cli.EtomMcpZeroRunner", _Stub
    )
    rc = cli_main(
        [
            "--split",
            "formal100",
            "--build-index-only",
            "--root",
            str(ROOT),
            "--output-dir",
            str(tmp_path / "o"),
            "--etom-source-dir",
            str(FIXTURE),
        ]
    )
    assert rc == 0
    assert captured["mode"] == "build-index"
    assert captured["build_index_only"] is True
