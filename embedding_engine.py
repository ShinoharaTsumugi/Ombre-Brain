# ============================================================
# Module: Embedding Engine (embedding_engine.py)
# 模块：向量化引擎
#
# Generates embeddings via Gemini API (OpenAI-compatible),
# stores them in SQLite, and provides cosine similarity search.
# 通过 Gemini API（OpenAI 兼容）生成 embedding，
# 存储在 SQLite 中，提供余弦相似度搜索。
#
# Depended on by: server.py, bucket_manager.py
# 被谁依赖：server.py, bucket_manager.py
# ============================================================

import os
import json
import math
import sqlite3
import logging

from openai import AsyncOpenAI

# numpy 在 requirements 里(导入模式/聚类已用);缺失时退化到纯 Python 逐条余弦,
# 功能不变只是慢——不让向量检索绑死在一个数值库上。
try:
    import numpy as _np
except ImportError:  # pragma: no cover
    _np = None

logger = logging.getLogger("ombre_brain.embedding")


class EmbeddingEngine:
    """
    Embedding generation + SQLite vector storage + cosine search.
    向量生成 + SQLite 向量存储 + 余弦搜索。
    """

    def __init__(self, config: dict):
        dehy_cfg = config.get("dehydration", {})
        embed_cfg = config.get("embedding", {})

        self.api_key = (embed_cfg.get("api_key") or dehy_cfg.get("api_key") or "").strip()
        self.base_url = (
            (embed_cfg.get("base_url") or "").strip()
            or (dehy_cfg.get("base_url") or "").strip()
            or "https://generativelanguage.googleapis.com/v1beta/openai/"
        )
        self.model = embed_cfg.get("model", "gemini-embedding-001")
        self.enabled = bool(self.api_key) and embed_cfg.get("enabled", True)

        # --- SQLite path: buckets_dir/embeddings.db ---
        db_path = os.path.join(config["buckets_dir"], "embeddings.db")
        self.db_path = db_path

        # --- Initialize client ---
        if self.enabled:
            self.client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=30.0,
            )
        else:
            self.client = None

        # --- 内存向量缓存(2026-08-13 breath 提速三轮) ---
        # search_similar 原本每次调用都把全表(~400 行 × 3072 维)的 JSON 文本从
        # sqlite 读出并逐行反序列化(~24MB 文本解析),每次 breath 还要调两次
        # (搜索预筛 + 向量通道)——相位账单里 vector≈1.6s 的真身,并同时抬高 search。
        # 改为首次使用时全量加载一次(float32 ~5MB),之后 store/delete 同步维护;
        # 有 numpy 时懒堆成归一化矩阵,余弦=一次矩阵点积。单容器部署,外部脚本
        # 直接改 embeddings.db 的场景(一次性维护)重启后自然重建。
        self._vec_cache: dict | None = None   # bucket_id -> 向量(numpy 时 float32 ndarray,否则 list)
        self._matrix = None                   # 归一化矩阵(numpy 路径,懒建)
        self._matrix_ids: list[str] = []      # 矩阵行对应的 bucket_id
        self._matrix_dirty = True

        # --- Initialize SQLite ---
        self._init_db()

    def _init_db(self):
        """Create embeddings table if not exists."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                bucket_id TEXT PRIMARY KEY,
                embedding TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        conn.commit()
        conn.close()

    async def generate_and_store(self, bucket_id: str, content: str) -> bool:
        """
        Generate embedding for content and store in SQLite.
        为内容生成 embedding 并存入 SQLite。
        Returns True on success, False on failure.
        """
        if not self.enabled or not content or not content.strip():
            return False

        try:
            embedding = await self._generate_embedding(content)
            if not embedding:
                return False
            self._store_embedding(bucket_id, embedding)
            return True
        except Exception as e:
            logger.warning(f"Embedding generation failed for {bucket_id}: {e}")
            return False

    async def _generate_embedding(self, text: str) -> list[float]:
        """Call API to generate embedding vector."""
        # Truncate to avoid token limits
        truncated = text[:2000]
        try:
            response = await self.client.embeddings.create(
                model=self.model,
                input=truncated,
            )
            if response.data and len(response.data) > 0:
                return response.data[0].embedding
            return []
        except Exception as e:
            logger.warning(f"Embedding API call failed: {e}")
            return []

    def _as_cache_vec(self, embedding: list[float]):
        """缓存内的向量表示:有 numpy 存 float32 数组(~1/8 内存),否则原样 list。"""
        if _np is not None:
            return _np.asarray(embedding, dtype=_np.float32)
        return list(embedding)

    def _load_vec_cache(self) -> None:
        """首次使用时把全表向量反序列化进内存;之后 store/delete 同步维护。"""
        if self._vec_cache is not None:
            return
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute("SELECT bucket_id, embedding FROM embeddings").fetchall()
        conn.close()
        cache = {}
        for bucket_id, emb_json in rows:
            try:
                cache[bucket_id] = self._as_cache_vec(json.loads(emb_json))
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
        self._vec_cache = cache
        self._matrix_dirty = True

    def _ensure_matrix(self, dim: int) -> None:
        """
        把向量缓存懒堆成行归一化矩阵(numpy 路径)。维度不匹配的向量跳过——
        旧逐条实现里它们余弦记 0 分、从不入选,语义等价;零向量归一化除 1
        (该行点积恒 0,同样等价)。
        """
        if (not self._matrix_dirty and self._matrix is not None
                and self._matrix.shape[1] == dim):
            return
        ids, rows = [], []
        for bucket_id, vec in self._vec_cache.items():
            if len(vec) != dim:
                continue
            ids.append(bucket_id)
            rows.append(vec)
        if not rows:
            self._matrix = None
            self._matrix_ids = []
            self._matrix_dirty = False
            return
        m = _np.vstack(rows).astype(_np.float32, copy=False)
        norms = _np.linalg.norm(m, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._matrix = m / norms
        self._matrix_ids = ids
        self._matrix_dirty = False

    def _store_embedding(self, bucket_id: str, embedding: list[float]):
        """Store embedding in SQLite (+ keep the in-memory cache in sync)."""
        from utils import now_iso
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT OR REPLACE INTO embeddings (bucket_id, embedding, updated_at) VALUES (?, ?, ?)",
            (bucket_id, json.dumps(embedding), now_iso()),
        )
        conn.commit()
        conn.close()
        if self._vec_cache is not None:
            self._vec_cache[bucket_id] = self._as_cache_vec(embedding)
            self._matrix_dirty = True

    def delete_embedding(self, bucket_id: str):
        """Remove embedding when bucket is deleted (+ cache sync)."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("DELETE FROM embeddings WHERE bucket_id = ?", (bucket_id,))
        conn.commit()
        conn.close()
        if self._vec_cache is not None and self._vec_cache.pop(bucket_id, None) is not None:
            self._matrix_dirty = True

    async def get_embedding(self, bucket_id: str) -> list[float] | None:
        """Retrieve stored embedding for a bucket. Returns None if not found."""
        self._load_vec_cache()
        vec = self._vec_cache.get(bucket_id)
        if vec is None:
            return None
        return vec.tolist() if _np is not None else list(vec)

    async def search_similar(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """
        Search for buckets similar to query text.
        Returns list of (bucket_id, similarity_score) sorted by score desc.
        搜索与查询文本相似的桶。返回 (bucket_id, 相似度分数) 列表。
        """
        if not self.enabled:
            return []

        try:
            # query 嵌入缓存:同一次 breath 里 query 会被嵌入两次(搜索预筛+向量通道),
            # 跨轮重复 query 也命中;嵌入确定性,语义零变化。
            cache = getattr(self, "_query_emb_cache", None)
            if cache is None:
                cache = self._query_emb_cache = {}
            query_embedding = cache.get(query)
            if query_embedding is None:
                query_embedding = await self._generate_embedding(query)
                if query_embedding:
                    if len(cache) >= 64:
                        cache.clear()
                    cache[query] = query_embedding
            if not query_embedding:
                return []
        except Exception as e:
            logger.warning(f"Query embedding failed: {e}")
            return []

        # 内存缓存代替旧的「每次全表 SELECT + 逐行 json.loads」——
        # 那是 breath 相位账单里 vector≈1.6s 的主体(每次 breath 还调两次)。
        self._load_vec_cache()
        if not self._vec_cache:
            return []

        if _np is not None:
            q = _np.asarray(query_embedding, dtype=_np.float32)
            qn = float(_np.linalg.norm(q))
            if qn == 0.0:
                return []
            self._ensure_matrix(len(query_embedding))
            if self._matrix is None:
                return []
            sims = self._matrix @ (q / qn)
            order = _np.argsort(sims)[::-1][:top_k]
            return [(self._matrix_ids[i], float(sims[i])) for i in order]

        # 纯 Python 退化路径:仍逐条余弦,但至少免去了反序列化大头
        results = []
        for bucket_id, stored_embedding in self._vec_cache.items():
            sim = self._cosine_similarity(query_embedding, stored_embedding)
            results.append((bucket_id, sim))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if len(a) != len(b) or not a:
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
