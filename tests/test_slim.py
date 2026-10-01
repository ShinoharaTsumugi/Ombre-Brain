# ============================================================
# Test: memory slimming (2026-10-02)
# 测试：瘦身改动——向量缓存逐行加载、矩阵相似度、关系图取邻居、dream 结晶提示、/health 字段
#
# All against a temp vault + synthetic vectors. No network, no real data.
# 全部在临时目录 + 合成向量上跑，不联网、不碰真实记忆。
# ============================================================

import importlib
import json
import os
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pytest
import pytest_asyncio

ROOT = Path(__file__).resolve().parent.parent
DIM = 64


@pytest.fixture(scope="module")
def srv(tmp_path_factory):
    """Import server.py against a temp buckets dir with embeddings 'enabled' (dummy key, never called)."""
    tmp = tmp_path_factory.mktemp("slim")
    bd = str(tmp / "buckets")
    for d in ["permanent", "dynamic", "archive", "feel"]:
        os.makedirs(os.path.join(bd, d), exist_ok=True)
    os.environ["OMBRE_BUCKETS_DIR"] = bd
    os.environ["OMBRE_API_KEY"] = "test-key-never-used"
    os.environ["OMBRE_HOOK_URL"] = ""
    os.environ.pop("OMBRE_DASHBOARD_PASSWORD", None)
    sys.path.insert(0, str(ROOT))
    for m in ("server", "utils", "bucket_manager", "embedding_engine", "dehydrator", "decay_engine", "import_memory"):
        sys.modules.pop(m, None)
    server = importlib.import_module("server")
    assert server.embedding_engine.enabled
    return server


def _unit(rng, dim=DIM):
    v = rng.normal(size=dim)
    return v / np.linalg.norm(v)


def _cluster(rng, base, noise=0.025):
    v = base + rng.normal(size=base.shape) * noise
    return v / np.linalg.norm(v)


async def _mk(server, content, kind="dynamic", vec=None):
    bid = await server.bucket_mgr.create(
        content=content, tags=["测试"], importance=5,
        domain=[] if kind == "feel" else ["测试"],
        valence=0.5, arousal=0.3, name=None, bucket_type=kind,
    )
    if vec is not None:
        server.embedding_engine._store_embedding(bid, [float(x) for x in vec])
    return bid


@pytest_asyncio.fixture(scope="module")
async def world(srv):
    """3 clusters × 3 dynamic buckets + 3 singleton dynamic + 3 similar feels + 2 lone feels."""
    rng = np.random.default_rng(7)
    ee = srv.embedding_engine
    clusters, singles = [], []
    for c in range(3):
        base = _unit(rng)
        ids = [await _mk(srv, f"簇{c}的记忆{i}", vec=_cluster(rng, base)) for i in range(3)]
        clusters.append(ids)
    for i in range(3):
        singles.append(await _mk(srv, f"孤立记忆{i}", vec=_unit(rng)))
    fbase = _unit(rng)
    feels_sim = [await _mk(srv, f"相似的feel{i}", kind="feel", vec=_cluster(rng, fbase)) for i in range(3)]
    feels_lone = [await _mk(srv, f"独自的feel{i}", kind="feel", vec=_unit(rng)) for i in range(2)]
    ee._vec_cache = None  # force a fresh load in tests
    return dict(clusters=clusters, singles=singles, feels_sim=feels_sim, feels_lone=feels_lone)


# ------------------------------------------------------------
# _load_vec_cache: streaming load equals the old fetchall load, skips bad rows
# ------------------------------------------------------------
def test_vec_cache_streaming_matches_fetchall(srv, world):
    ee = srv.embedding_engine
    conn = sqlite3.connect(ee.db_path)
    conn.execute("INSERT OR REPLACE INTO embeddings VALUES (?, ?, ?)", ("badrow000000", "not json", "x"))
    conn.commit()
    rows = conn.execute("SELECT bucket_id, embedding FROM embeddings").fetchall()
    conn.close()
    expected = {}
    for bid, emb in rows:
        try:
            expected[bid] = np.asarray(json.loads(emb), dtype=np.float32)
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
    ee._vec_cache = None
    ee._load_vec_cache()
    assert set(ee._vec_cache) == set(expected)
    assert "badrow000000" not in ee._vec_cache
    for bid, vec in expected.items():
        assert np.array_equal(ee._vec_cache[bid], vec)
    assert ee._matrix_dirty is True


# ------------------------------------------------------------
# similarity_matrix: numpy path equals pure-Python cosine; order kept; unknown ids dropped
# ------------------------------------------------------------
def test_similarity_matrix_matches_cosine(srv, world):
    ee = srv.embedding_engine
    ids = world["clusters"][0] + world["singles"][:2]
    present, sims = ee.similarity_matrix(ids + ["000000000000"])
    assert present == ids
    assert sims.shape == (len(ids), len(ids))
    for i, a in enumerate(ids):
        for j, b in enumerate(ids):
            ref = ee._cosine_similarity(ee._vec_cache[a].tolist(), ee._vec_cache[b].tolist())
            assert abs(float(sims[i, j]) - ref) < 1e-4
    assert all(abs(float(sims[i, i]) - 1.0) < 1e-5 for i in range(len(ids)))
    assert float(sims[0, 1]) > 0.9           # same cluster
    assert float(sims[0, 3]) < 0.5           # cluster vs singleton


def test_similarity_matrix_degrades_cleanly(srv, world):
    ee = srv.embedding_engine
    assert ee.similarity_matrix([]) == ([], None)
    assert ee.similarity_matrix(["000000000000"]) == ([], None)
    enabled = ee.enabled
    try:
        ee.enabled = False
        assert ee.similarity_matrix(world["singles"]) == ([], None)
    finally:
        ee.enabled = enabled


# ------------------------------------------------------------
# _network_edges: exactly the intra-cluster pairs, deduped, honours min_sim and k
# ------------------------------------------------------------
def _pairs(edges):
    return {tuple(sorted((e["source"], e["target"]))) for e in edges}


def test_network_edges_clusters(srv, world):
    ids = [b for c in world["clusters"] for b in c] + world["singles"]
    edges = srv._network_edges(ids, 0.85, 6)
    expected = set()
    for c in world["clusters"]:
        for i in range(3):
            for j in range(i + 1, 3):
                expected.add(tuple(sorted((c[i], c[j]))))
    assert _pairs(edges) == expected
    assert len(edges) == len(expected)                      # no duplicates
    assert all(0.85 <= e["similarity"] <= 1.0 for e in edges)
    assert all(e["source"] != e["target"] for e in edges)


def test_network_edges_k_and_threshold(srv, world):
    ids = [b for c in world["clusters"] for b in c] + world["singles"]
    assert srv._network_edges(ids, 0.999, 6) == []           # nothing that similar
    k1 = srv._network_edges(ids, 0.85, 1)                    # one neighbour per node
    assert 0 < len(k1) <= len(ids)
    assert _pairs(k1) <= _pairs(srv._network_edges(ids, 0.85, 6))
    assert srv._network_edges(ids[:1], 0.0, 6) == []         # <2 nodes
    assert srv._network_edges([], 0.0, 6) == []


# ------------------------------------------------------------
# /api/network handler: nodes = live buckets, edges from _network_edges, params sanitised
# ------------------------------------------------------------
class _Req:
    def __init__(self, **q):
        self.query_params = q
        self.cookies = {}
        self.headers = {}


@pytest.mark.asyncio
async def test_api_network_handler(srv, world, monkeypatch):
    monkeypatch.setattr(srv, "_require_auth", lambda request: None)
    resp = await srv.api_network(_Req(min_sim="0.85", k="6"))
    body = json.loads(resp.body)
    live = await srv.bucket_mgr.list_all(include_archive=False)
    assert {n["id"] for n in body["nodes"]} == {b["id"] for b in live}
    expected = set()
    for c in world["clusters"]:
        for i in range(3):
            for j in range(i + 1, 3):
                expected.add(tuple(sorted((c[i], c[j]))))
    for i in range(3):
        for j in range(i + 1, 3):
            expected.add(tuple(sorted((world["feels_sim"][i], world["feels_sim"][j]))))
    assert _pairs(body["edges"]) == expected
    # garbage params fall back to defaults instead of 500
    resp2 = await srv.api_network(_Req(min_sim="abc", k="zzz"))
    assert resp2.status_code == 200
    assert _pairs(json.loads(resp2.body)["edges"]) == expected


# ------------------------------------------------------------
# dream(): crystallization hint via the matrix path and via the pure-Python fallback
# ------------------------------------------------------------
@pytest.mark.asyncio
async def test_dream_crystal_hint_both_paths(srv, world, monkeypatch):
    out = await srv.dream()
    assert "🔮 你已经写过 3 条相似的 feel" in out, out[-400:]
    ee = srv.embedding_engine
    monkeypatch.setattr(ee, "similarity_matrix", lambda ids: ([], None))
    out2 = await srv.dream()
    assert "🔮 你已经写过 3 条相似的 feel" in out2, out2[-400:]
    await srv.decay_engine.stop()


# ------------------------------------------------------------
# /health: rss / uptime / vec cache rows are reported
# ------------------------------------------------------------
@pytest.mark.asyncio
async def test_health_fields(srv, world):
    resp = await srv.health_check(None)
    body = json.loads(resp.body)
    assert body["status"] == "ok"
    assert body["uptime_s"] >= 0
    assert body["vec_cache_rows"] == len(srv.embedding_engine._vec_cache)
    if sys.platform.startswith("linux"):
        assert body["rss_mb"] and body["rss_mb"] > 10
