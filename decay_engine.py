# ============================================================
# Module: Memory Decay Engine (decay_engine.py)
# 模块：记忆衰减引擎
#
# Simulates human forgetting curve; auto-decays inactive memories and archives them.
# 模拟人类遗忘曲线，自动衰减不活跃记忆并归档。
#
# Core formula (improved Ebbinghaus + emotion coordinates):
# 核心公式（改进版艾宾浩斯遗忘曲线 + 情感坐标）：
#   Score = Importance × (activation_count^0.3) × e^(-λ×days) × emotion_weight
#
# Emotion weight (continuous coordinate, not discrete labels):
# 情感权重（基于连续坐标而非离散列举）：
#   emotion_weight = base + (arousal * arousal_boost) + (abs(valence - 0.5) * valence_boost)
#   Higher arousal → higher emotion weight → slower decay
#   唤醒度越高 → 情感权重越大 → 记忆衰减越慢
#
# Depended on by: server.py
# 被谁依赖：server.py
# ============================================================

import os
import math
import asyncio
import logging
from datetime import datetime

from utils import now_diff_days, now_iso

logger = logging.getLogger("ombre_brain.decay")

# ── 归档安全阀 ────────────────────────────────────────────────────────
# 2026-07 之前 days_since 因 naive/aware 相减异常恒为 30，未解决的桶分数恒在 1.3 上下，
# 永远跌不破阈值——「按时间遗忘」这条路径其实从未真正生效过（只有已解决的桶会被归档）。
# 现在时间维度修好了，若直接放开，积累两个月的旧桶会在一轮里被成批归档
# （λ=0.05、阈值 0.3 时，重要度5的桶约 59 天、重要度10的约 73 天就跌破）。
# 那是不可替代的个人记忆，不能因为一个 bug 修复被悄悄吞掉。
# 因此默认维持修复前的实际行为：只归档「已解决」的桶。
# 想启用完整的时间遗忘，设 OMBRE_DECAY_ARCHIVE_UNRESOLVED=true。
ARCHIVE_UNRESOLVED = os.environ.get(
    "OMBRE_DECAY_ARCHIVE_UNRESOLVED", ""
).strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default


# ── 2026-10-11 沉底 / 归档规则参数（全部可用环境变量调）───────────────────
# 背景：importance 写入后永不自动降低，而旧的自动结案只认 importance≤4，全库 1476 个
# 动态桶只有 36 个够得着；5 分以上的桶没有任何自动沉底的路。现在：
#   ① 低重要度闲置：importance ≤ AUTO_RESOLVE_MAX_IMPORTANCE 且闲置 > AUTO_RESOLVE_IDLE_DAYS 天
#   ② 事件过期：kind=event 且 until 已过 EVENT_UNTIL_GRACE_DAYS 天
#   ③ 待办过期：kind=todo 且 until 已过 TODO_UNTIL_GRACE_DAYS 天、且这期间没人碰
#   ④ 没写 until 的旧事件：kind=event、创建满 EVENT_NO_UNTIL_AGE_DAYS 天且闲置 ≥ EVENT_NO_UNTIL_IDLE_DAYS 天
# 结案只是沉底（不浮现、检索降权），不是消失：
#   ⑤ 归档宽限：结案后至少 RESOLVED_ARCHIVE_GRACE_DAYS 天内仍可被关键词搜到，之后分数低于阈值才归档
#   ⑥ 里程碑：importance ≥ ARCHIVE_NEVER_MIN_IMPORTANCE 的桶结案后只沉底，永不被引擎归档
AUTO_RESOLVE_MAX_IMPORTANCE = _env_int("OMBRE_AUTO_RESOLVE_MAX_IMPORTANCE", 5)
AUTO_RESOLVE_IDLE_DAYS = _env_int("OMBRE_AUTO_RESOLVE_IDLE_DAYS", 30)
EVENT_UNTIL_GRACE_DAYS = _env_int("OMBRE_EVENT_GRACE_DAYS", 14)
TODO_UNTIL_GRACE_DAYS = _env_int("OMBRE_TODO_GRACE_DAYS", 45)
EVENT_NO_UNTIL_AGE_DAYS = _env_int("OMBRE_EVENT_AGE_DAYS", 60)
EVENT_NO_UNTIL_IDLE_DAYS = _env_int("OMBRE_EVENT_IDLE_DAYS", 30)
RESOLVED_ARCHIVE_GRACE_DAYS = _env_int("OMBRE_RESOLVED_GRACE_DAYS", 30)
ARCHIVE_NEVER_MIN_IMPORTANCE = _env_int("OMBRE_ARCHIVE_NEVER_IMPORTANCE", 9)


class DecayEngine:
    """
    Memory decay engine — periodically scans all dynamic buckets,
    calculates decay scores, auto-archives low-activity buckets
    to simulate natural forgetting.
    记忆衰减引擎 —— 定期扫描所有动态桶，
    计算衰减得分，将低活跃桶自动归档，模拟自然遗忘。
    """

    def __init__(self, config: dict, bucket_mgr):
        # --- Load decay parameters / 加载衰减参数 ---
        decay_cfg = config.get("decay", {})
        self.decay_lambda = decay_cfg.get("lambda", 0.05)
        self.threshold = decay_cfg.get("threshold", 0.3)
        self.check_interval = decay_cfg.get("check_interval_hours", 24)

        # --- Emotion weight params (continuous arousal coordinate) ---
        # --- 情感权重参数（基于连续 arousal 坐标）---
        emotion_cfg = decay_cfg.get("emotion_weights", {})
        self.emotion_base = emotion_cfg.get("base", 1.0)
        self.arousal_boost = emotion_cfg.get("arousal_boost", 0.8)
        self.valence_boost = emotion_cfg.get("valence_boost", 0.3)

        self.bucket_mgr = bucket_mgr

        # --- Background task control / 后台任务控制 ---
        self._task: asyncio.Task | None = None
        self._running = False

    @property
    def is_running(self) -> bool:
        """Whether the decay engine is running in the background.
        衰减引擎是否正在后台运行。"""
        return self._running

    # ---------------------------------------------------------
    # Core: calculate decay score for a single bucket
    # 核心：计算单个桶的衰减得分
    #
    # Higher score = more vivid memory; below threshold → archive
    # 得分越高 = 记忆越鲜活，低于阈值则归档
    # Permanent buckets never decay / 固化桶永远不衰减
    # ---------------------------------------------------------
    # ---------------------------------------------------------
    # Freshness bonus: continuous exponential decay
    # 新鲜度加成：连续指数衰减
    # bonus = 1.0 + 1.0 × e^(-t/36), t in hours
    # t=0 → 2.0×, t≈25h(半衰) → 1.5×, t≈72h → ≈1.14×, t→∞ → 1.0×
    # ---------------------------------------------------------
    @staticmethod
    def _calc_time_weight(days_since: float) -> float:
        """
        Freshness bonus multiplier: 1.0 + e^(-t/36), t in hours.
        新鲜度加成乘数：刚存入×2.0，~36小时半衰，72小时后趋近×1.0。
        """
        hours = days_since * 24.0
        return 1.0 + 1.0 * math.exp(-hours / 36.0)

    def calculate_score(self, metadata: dict) -> float:
        """
        Calculate current activity score for a memory bucket.
        计算一个记忆桶的当前活跃度得分。

        New model: short-term vs long-term weight separation.
        新模型：短期/长期权重分离。
        - Short-term (≤3 days): time_weight dominates, emotion amplifies
        - Long-term (>3 days): emotion_weight dominates, time decays to floor
        短期（≤3天）：时间权重主导，情感放大
        长期（>3天）：情感权重主导，时间衰减到底线
        """
        if not isinstance(metadata, dict):
            return 0.0

        # --- Pinned/protected buckets: never decay, importance locked to 10 ---
        if metadata.get("pinned") or metadata.get("protected"):
            return 999.0

        # --- Permanent buckets never decay ---
        if metadata.get("type") == "permanent":
            return 999.0

        # --- Feel buckets: never decay, fixed moderate score ---
        if metadata.get("type") == "feel":
            return 50.0

        importance = max(1, min(10, int(metadata.get("importance", 5))))
        activation_count = max(1.0, float(metadata.get("activation_count", 1)))

        # --- Days since last activation ---
        # 时间戳带时区偏移，必须用 aware 的当前时间去减（历史上这里用朴素 datetime.now()，
        # 相减必抛 TypeError 被吞掉 → days_since 恒为 30，衰减的时间维度形同虚设）
        days_since = now_diff_days(metadata.get("last_active", metadata.get("created", "")))
        if days_since is None:
            days_since = 30
        days_since = max(0.0, days_since)

        # --- Emotion weight ---
        try:
            arousal = max(0.0, min(1.0, float(metadata.get("arousal", 0.3))))
        except (ValueError, TypeError):
            arousal = 0.3
        try:
            valence = max(0.0, min(1.0, float(metadata.get("valence", 0.5))))
        except (ValueError, TypeError):
            valence = 0.5
        emotion_weight = self.emotion_base + arousal * self.arousal_boost + abs(valence - 0.5) * self.valence_boost

        # --- Time weight ---
        time_weight = self._calc_time_weight(days_since)

        # --- Short-term vs Long-term weight separation ---
        # 短期（≤3天）：time_weight 占 70%，emotion 占 30%
        # 长期（>3天）：emotion 占 70%，time_weight 占 30%
        if days_since <= 3.0:
            # Short-term: time dominates, emotion amplifies
            combined_weight = time_weight * 0.7 + emotion_weight * 0.3
        else:
            # Long-term: emotion dominates, time provides baseline
            combined_weight = emotion_weight * 0.7 + time_weight * 0.3

        # --- Base score ---
        base_score = (
            importance
            * (activation_count ** 0.3)
            * math.exp(-self.decay_lambda * days_since)
            * combined_weight
        )

        # --- Weight pool modifiers ---
        # resolved + digested (has feel) → accelerated fade: ×0.02
        # resolved only → ×0.05
        # 已处理+已消化（写过feel）→ 加速淡化：×0.02
        # 仅已处理 → ×0.05
        resolved = metadata.get("resolved", False)
        digested = metadata.get("digested", False)  # set when feel is written for this memory
        if resolved and digested:
            resolved_factor = 0.02
        elif resolved:
            resolved_factor = 0.05
        else:
            resolved_factor = 1.0
        urgency_boost = 1.5 if (arousal > 0.7 and not resolved) else 1.0

        return round(base_score * resolved_factor * urgency_boost, 4)

    # ---------------------------------------------------------
    # Execute one decay cycle
    # 执行一轮衰减周期
    # Scan all dynamic buckets → score → archive those below threshold
    # 扫描所有动态桶 → 算分 → 低于阈值的归档
    # ---------------------------------------------------------
    def auto_resolve_reason(self, meta: dict):
        """
        Should this (unresolved, dynamic) bucket be auto-resolved? Return the reason, or None.
        这个（未解决的动态）桶该不该自动结案？返回原因字符串，不该返回 None。
        规则见文件头 ①~④；调用方负责只传未解决、非钉选、非 feel 的桶。
        """
        try:
            imp = int(meta.get("importance", 5) or 5)
        except (TypeError, ValueError):
            imp = 5
        idle = now_diff_days(meta.get("last_active", meta.get("created", "")))
        idle = 999.0 if idle is None else max(0.0, idle)
        kind = str(meta.get("kind") or "").strip().lower()
        until = str(meta.get("until") or "").strip()[:10]
        until_past = now_diff_days(until) if until else None     # until 已过去几天（负数=还没到）

        # ① 低重要度闲置
        if imp <= AUTO_RESOLVE_MAX_IMPORTANCE and idle > AUTO_RESOLVE_IDLE_DAYS:
            return f"低重要度闲置 imp={imp} idle={idle:.0f}d"
        # ② 事件过期 / ③ 待办过期（待办还要求这期间没人碰）
        if kind in ("event", "todo") and until_past is not None:
            grace = EVENT_UNTIL_GRACE_DAYS if kind == "event" else TODO_UNTIL_GRACE_DAYS
            if until_past > grace and (kind == "event" or idle > grace):
                return f"{kind} 已过期 until={until} +{until_past:.0f}d"
        # ④ 没写 until 的旧事件：创建够久 + 闲置够久
        if kind == "event" and not until:
            age = now_diff_days(meta.get("created", ""))
            age = 0.0 if age is None else age
            if age >= EVENT_NO_UNTIL_AGE_DAYS and idle >= EVENT_NO_UNTIL_IDLE_DAYS:
                return f"旧事件闲置 age={age:.0f}d idle={idle:.0f}d"
        return None

    async def run_decay_cycle(self) -> dict:
        """
        Execute one decay cycle: iterate dynamic buckets, archive those
        scoring below threshold.
        执行一轮衰减：遍历动态桶，归档得分低于阈值的桶。

        Returns stats: {"checked": N, "archived": N, "lowest_score": X}
        """
        try:
            buckets = await self.bucket_mgr.list_all(include_archive=False)
        except Exception as e:
            logger.error(f"Failed to list buckets for decay / 衰减周期列桶失败: {e}")
            return {"checked": 0, "archived": 0, "lowest_score": 0, "error": str(e)}

        checked = 0
        archived = 0
        auto_resolved = 0
        skipped_unresolved = 0   # 低于阈值但因安全阀保住的未解决桶
        kept_grace = 0           # 已结案但还在归档宽限期内
        kept_milestone = 0       # 已结案但重要度够高、永不归档
        lowest_score = float("inf")

        for bucket in buckets:
            meta = bucket.get("metadata", {})

            # Skip permanent / pinned / protected / feel buckets
            # 跳过固化桶、钉选/保护桶和 feel 桶
            if meta.get("type") in ("permanent", "feel") or meta.get("pinned") or meta.get("protected"):
                continue

            checked += 1

            # --- Auto-resolve (rules ①~④, see auto_resolve_reason) ---
            # --- 自动结案（规则 ①~④，见 auto_resolve_reason）---
            # _no_touch：结案这个动作本身不能刷新 last_active，否则桶又变新鲜了
            if not meta.get("resolved", False):
                reason = self.auto_resolve_reason(meta)
                if reason:
                    try:
                        await self.bucket_mgr.update(bucket["id"], resolved=True, _no_touch=True)
                        meta["resolved"] = True  # refresh local meta so resolved_factor applies this cycle
                        meta.setdefault("resolved_at", now_iso())
                        auto_resolved += 1
                        logger.info(
                            f"Auto-resolved / 自动结案: "
                            f"{meta.get('name', bucket['id'])} ({reason})"
                        )
                    except Exception as e:
                        logger.warning(f"Auto-resolve failed / 自动结案失败: {e}")

            try:
                score = self.calculate_score(meta)
            except Exception as e:
                logger.warning(
                    f"Score calculation failed for {bucket.get('id', '?')} / "
                    f"计算得分失败: {e}"
                )
                continue

            lowest_score = min(lowest_score, score)

            # --- Below threshold → archive (simulate forgetting) ---
            # --- 低于阈值 → 归档（模拟遗忘）---
            if score < self.threshold:
                if meta.get("resolved", False):
                    # ⑥ 里程碑：重要度够高的结案桶只沉底，永不归档
                    try:
                        _imp = int(meta.get("importance", 5) or 5)
                    except (TypeError, ValueError):
                        _imp = 5
                    if _imp >= ARCHIVE_NEVER_MIN_IMPORTANCE:
                        kept_milestone += 1
                        continue
                    # ⑤ 归档宽限：结案后至少保留可搜 N 天。老桶没有 resolved_at 就先盖一个（从今天起算）
                    _r_at = meta.get("resolved_at")
                    if not _r_at:
                        try:
                            await self.bucket_mgr.update(bucket["id"], resolved=True, _no_touch=True)
                        except Exception as ex:
                            logger.warning(f"resolved_at stamp failed / 补结案时间戳失败: {ex}")
                        kept_grace += 1
                        continue
                    _r_days = now_diff_days(_r_at)
                    if _r_days is None or _r_days < RESOLVED_ARCHIVE_GRACE_DAYS:
                        kept_grace += 1
                        continue
                elif not ARCHIVE_UNRESOLVED:
                    # 安全阀：未解决的桶默认不因时间衰减被归档（见文件头说明）
                    skipped_unresolved += 1
                    continue
                try:
                    success = await self.bucket_mgr.archive(bucket["id"])
                    if success:
                        archived += 1
                        logger.info(
                            f"Decay archived / 衰减归档: "
                            f"{meta.get('name', bucket['id'])} "
                            f"(score={score:.4f}, threshold={self.threshold})"
                        )
                except Exception as e:
                    logger.warning(
                        f"Archive failed for {bucket.get('id', '?')} / "
                        f"归档失败: {e}"
                    )

        result = {
            "checked": checked,
            "archived": archived,
            "auto_resolved": auto_resolved,
            "protected_unresolved": skipped_unresolved,
            "kept_grace": kept_grace,
            "kept_milestone": kept_milestone,
            "lowest_score": lowest_score if checked > 0 else 0,
        }
        logger.info(f"Decay cycle complete / 衰减周期完成: {result}")
        if skipped_unresolved:
            logger.info(
                f"{skipped_unresolved} 个未解决桶已跌破阈值但被安全阀保留"
                f"（要启用按时间遗忘：OMBRE_DECAY_ARCHIVE_UNRESOLVED=true）"
            )
        return result

    # ---------------------------------------------------------
    # Background decay task management
    # 后台衰减任务管理
    # ---------------------------------------------------------
    async def ensure_started(self) -> None:
        """
        Ensure the decay engine is started (lazy init on first call).
        确保衰减引擎已启动（懒加载，首次调用时启动）。
        """
        if not self._running:
            await self.start()

    async def start(self) -> None:
        """Start the background decay loop.
        启动后台衰减循环。"""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._background_loop())
        logger.info(
            f"Decay engine started, interval: {self.check_interval}h / "
            f"衰减引擎已启动，检查间隔: {self.check_interval} 小时"
        )

    async def stop(self) -> None:
        """Stop the background decay loop.
        停止后台衰减循环。"""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Decay engine stopped / 衰减引擎已停止")

    async def _background_loop(self) -> None:
        """Background loop: run decay → sleep → repeat.
        后台循环体：执行衰减 → 睡眠 → 重复。"""
        while self._running:
            try:
                await self.run_decay_cycle()
            except Exception as e:
                logger.error(f"Decay cycle error / 衰减周期出错: {e}")
            # --- Wait for next cycle / 等待下一个周期 ---
            try:
                await asyncio.sleep(self.check_interval * 3600)
            except asyncio.CancelledError:
                break
