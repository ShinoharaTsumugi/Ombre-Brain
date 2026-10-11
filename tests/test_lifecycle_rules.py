# ============================================================
# Test: memory lifecycle rules (2026-10-11) — pure local, no LLM
# 测试：记忆生命周期规则 —— 纯本地，不需要 LLM
#
#   - auto-resolve threshold importance ≤ 5 (was 4)
#   - kind=event + until expired → auto-resolve
#   - kind=todo + until expired requires idleness
#   - old event without until → auto-resolve by age + idleness
#   - resolved buckets keep a grace window before archive
#   - importance ≥ 9 resolved buckets are never archived
#   - update(): kind/until round-trip, resolved_at stamping, _no_touch
#   - dehydrator parsers: kind/until/importance clamp
# ============================================================
import re
import pytest
import pytest_asyncio
import frontmatter as fm
from datetime import datetime, timedelta, timezone

from utils import LOCAL_TZ


def _iso(days_ago: float) -> str:
    return (datetime.now(LOCAL_TZ) - timedelta(days=days_ago)).isoformat()


def _date(days_ago: int) -> str:
    return (datetime.now(LOCAL_TZ) - timedelta(days=days_ago)).strftime("%Y-%m-%d")


async def _mk(bucket_mgr, content="一条测试记忆，内容足够长一点点", **fields):
    """create() then patch frontmatter fields directly (created/last_active/resolved/...)."""
    create_kw = {k: fields.pop(k) for k in list(fields) if k in ("importance", "kind", "until", "bucket_type", "pinned")}
    bid = await bucket_mgr.create(content=content, tags=["测试"], domain=["日常"], **create_kw)
    if fields:
        path = bucket_mgr._find_bucket_file(bid)
        post = fm.load(path)
        for k, v in fields.items():
            post[k] = v
        with open(path, "w", encoding="utf-8") as f:
            f.write(fm.dumps(post))
        # invalidate module index cache so list_all sees patched metadata
        import bucket_manager as _bm
        _bm._index_cache["entries"] = None
    return bid


async def _meta(bucket_mgr, bid):
    b = await bucket_mgr.get(bid)
    return b["metadata"] if b else None


# ------------------------------------------------------------
# auto_resolve_reason
# ------------------------------------------------------------
class TestAutoResolveRules:
    def test_threshold_is_5(self, decay_eng):
        assert decay_eng.auto_resolve_reason({"importance": 5, "last_active": _iso(31)}) is not None
        assert decay_eng.auto_resolve_reason({"importance": 6, "last_active": _iso(31)}) is None
        assert decay_eng.auto_resolve_reason({"importance": 5, "last_active": _iso(10)}) is None

    def test_event_until_expired(self, decay_eng):
        base = {"importance": 8, "last_active": _iso(1), "created": _iso(40), "kind": "event"}
        assert decay_eng.auto_resolve_reason({**base, "until": _date(20)}) is not None   # 20d past > 14d grace
        assert decay_eng.auto_resolve_reason({**base, "until": _date(5)}) is None        # inside grace
        assert decay_eng.auto_resolve_reason({**base, "until": _date(-10)}) is None      # future

    def test_todo_until_requires_idle(self, decay_eng):
        base = {"importance": 7, "created": _iso(90), "kind": "todo", "until": _date(60)}
        assert decay_eng.auto_resolve_reason({**base, "last_active": _iso(5)}) is None      # touched recently
        assert decay_eng.auto_resolve_reason({**base, "last_active": _iso(50)}) is not None  # idle > 45d

    def test_old_event_without_until(self, decay_eng):
        base = {"importance": 8, "kind": "event"}
        assert decay_eng.auto_resolve_reason({**base, "created": _iso(70), "last_active": _iso(40)}) is not None
        assert decay_eng.auto_resolve_reason({**base, "created": _iso(70), "last_active": _iso(5)}) is None
        assert decay_eng.auto_resolve_reason({**base, "created": _iso(30), "last_active": _iso(30)}) is None

    def test_fact_never_auto_resolves_by_age(self, decay_eng):
        meta = {"importance": 8, "kind": "fact", "created": _iso(300), "last_active": _iso(200)}
        assert decay_eng.auto_resolve_reason(meta) is None


# ------------------------------------------------------------
# run_decay_cycle: resolve + grace + milestone
# ------------------------------------------------------------
class TestDecayCycle:
    @pytest.mark.asyncio
    async def test_cycle_resolves_and_stamps_without_touch(self, bucket_mgr, decay_eng):
        bid = await _mk(bucket_mgr, importance=5, created=_iso(40), last_active=_iso(35))
        before = (await _meta(bucket_mgr, bid))["last_active"]
        stats = await decay_eng.run_decay_cycle()
        m = await _meta(bucket_mgr, bid)
        assert stats["auto_resolved"] == 1
        assert m["resolved"] is True
        assert m.get("resolved_at")
        assert m["last_active"] == before          # _no_touch: 沉底不刷新 last_active

    @pytest.mark.asyncio
    async def test_resolved_kept_during_grace_then_archived(self, bucket_mgr, decay_eng):
        # score of an imp-5 bucket idle 120d, resolved → far below 0.3
        bid = await _mk(bucket_mgr, importance=5, created=_iso(150), last_active=_iso(120),
                        resolved=True, resolved_at=_iso(5))
        stats = await decay_eng.run_decay_cycle()
        assert stats["archived"] == 0 and stats["kept_grace"] == 1
        assert (await _meta(bucket_mgr, bid))["type"] == "dynamic"
        # push resolved_at past the grace window
        path = bucket_mgr._find_bucket_file(bid)
        post = fm.load(path); post["resolved_at"] = _iso(40)
        with open(path, "w", encoding="utf-8") as f:
            f.write(fm.dumps(post))
        import bucket_manager as _bm; _bm._index_cache["entries"] = None
        stats = await decay_eng.run_decay_cycle()
        assert stats["archived"] == 1
        assert (await _meta(bucket_mgr, bid))["type"] == "archived"

    @pytest.mark.asyncio
    async def test_legacy_resolved_gets_stamp_not_archive(self, bucket_mgr, decay_eng):
        bid = await _mk(bucket_mgr, importance=5, created=_iso(150), last_active=_iso(120), resolved=True)
        stats = await decay_eng.run_decay_cycle()
        m = await _meta(bucket_mgr, bid)
        assert stats["archived"] == 0 and stats["kept_grace"] == 1
        assert m.get("resolved_at") and m["type"] == "dynamic"

    @pytest.mark.asyncio
    async def test_milestone_never_archived(self, bucket_mgr, decay_eng):
        bid = await _mk(bucket_mgr, importance=9, created=_iso(300), last_active=_iso(250),
                        resolved=True, resolved_at=_iso(200))
        stats = await decay_eng.run_decay_cycle()
        assert stats["archived"] == 0 and stats["kept_milestone"] == 1
        assert (await _meta(bucket_mgr, bid))["type"] == "dynamic"

    @pytest.mark.asyncio
    async def test_unresolved_safety_valve_still_holds(self, bucket_mgr, decay_eng):
        # imp 7, idle 200d, no kind → no rule fires, score < 0.3, but unresolved → kept
        bid = await _mk(bucket_mgr, importance=7, created=_iso(220), last_active=_iso(200))
        stats = await decay_eng.run_decay_cycle()
        assert stats["auto_resolved"] == 0 and stats["archived"] == 0
        assert stats["protected_unresolved"] == 1
        assert (await _meta(bucket_mgr, bid))["type"] == "dynamic"


# ------------------------------------------------------------
# bucket_manager.update: kind / until / resolved_at / _no_touch
# ------------------------------------------------------------
class TestUpdateFields:
    @pytest.mark.asyncio
    async def test_create_and_update_kind_until(self, bucket_mgr):
        bid = await _mk(bucket_mgr, kind="event", until="2026-06-20")
        m = await _meta(bucket_mgr, bid)
        assert m["kind"] == "event" and m["until"] == "2026-06-20"
        assert await bucket_mgr.update(bid, kind="todo", until="2026-07-01")
        m = await _meta(bucket_mgr, bid)
        assert m["kind"] == "todo" and m["until"] == "2026-07-01"
        assert await bucket_mgr.update(bid, until="-", kind="-")
        m = await _meta(bucket_mgr, bid)
        assert "until" not in m and "kind" not in m

    @pytest.mark.asyncio
    async def test_invalid_kind_until_ignored(self, bucket_mgr):
        bid = await _mk(bucket_mgr, kind="whatever", until="next friday")
        m = await _meta(bucket_mgr, bid)
        assert "kind" not in m and "until" not in m

    @pytest.mark.asyncio
    async def test_resolved_at_stamp_and_clear(self, bucket_mgr):
        bid = await _mk(bucket_mgr)
        await bucket_mgr.update(bid, resolved=True)
        m = await _meta(bucket_mgr, bid)
        assert m["resolved"] is True and m.get("resolved_at")
        stamp = m["resolved_at"]
        await bucket_mgr.update(bid, resolved=True)          # re-resolve keeps the original stamp
        assert (await _meta(bucket_mgr, bid))["resolved_at"] == stamp
        await bucket_mgr.update(bid, resolved=False)
        assert "resolved_at" not in (await _meta(bucket_mgr, bid))

    @pytest.mark.asyncio
    async def test_no_touch_keeps_last_active(self, bucket_mgr):
        bid = await _mk(bucket_mgr, last_active=_iso(50))
        before = (await _meta(bucket_mgr, bid))["last_active"]
        await bucket_mgr.update(bid, resolved=True, _no_touch=True)
        assert (await _meta(bucket_mgr, bid))["last_active"] == before
        await bucket_mgr.update(bid, resolved=False)
        assert (await _meta(bucket_mgr, bid))["last_active"] != before


# ------------------------------------------------------------
# dehydrator parsers
# ------------------------------------------------------------
class TestDehydratorParsers:
    def _dh(self, test_config):
        from dehydrator import Dehydrator
        return Dehydrator(test_config)

    def test_parse_digest_fields(self, test_config):
        dh = self._dh(test_config)
        raw = ('[{"name":"考试","content":"六月二十日的日语考试考完了，考得还行，之后一起吃了饭庆祝一下",'
               '"domain":["考试"],"valence":0.7,"arousal":0.5,"tags":["考试"],"importance":12,"kind":"EVENT","until":"2026-06-20"},'
               '{"name":"口味","content":"她从小吃甜口清淡菜，长大后能吃微辣到中辣，不吃重辣的四川菜",'
               '"domain":["饮食"],"valence":0.5,"arousal":0.3,"tags":["口味"],"importance":7,"kind":"fact","until":null}]')
        items = dh._parse_digest(raw)
        assert items[0]["importance"] == 9 and items[0]["kind"] == "event" and items[0]["until"] == "2026-06-20"
        assert items[1]["kind"] == "fact" and items[1]["until"] is None

    def test_parse_analysis_fields(self, test_config):
        dh = self._dh(test_config)
        raw = ('{"domain":["计划"],"valence":0.6,"arousal":0.4,"tags":["行李"],"suggested_name":"回悉尼行李",'
               '"importance":"5","kind":"todo","until":"2026-07-30T00:00:00"}')
        a = dh._parse_analysis(raw)
        assert a["importance"] == 5 and a["kind"] == "todo" and a["until"] == "2026-07-30"
        a2 = dh._parse_analysis('{"domain":["日常"],"valence":0.5,"arousal":0.3,"tags":[],"suggested_name":"x","until":"next week","kind":"plan"}')
        assert a2["importance"] is None and a2["kind"] is None and a2["until"] is None

    def test_prompts_carry_rubric(self):
        import dehydrator
        assert "importance 刻度" in dehydrator.DIGEST_PROMPT
        assert "importance 刻度" in dehydrator.ANALYZE_PROMPT
        assert re.search(r"今天是 \d{4}-\d{2}-\d{2}", dehydrator._today_line())
