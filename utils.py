# ============================================================
# Module: Common Utilities (utils.py)
# 模块：通用工具函数
#
# Provides config loading, logging init, path safety, ID generation, etc.
# 提供配置加载、日志初始化、路径安全校验、ID 生成等基础能力
#
# Depended on by: server.py, bucket_manager.py, dehydrator.py, decay_engine.py
# 被谁依赖：server.py, bucket_manager.py, dehydrator.py, decay_engine.py
# ============================================================

import os
import re
import uuid
import yaml
import logging
from pathlib import Path
from datetime import datetime, timezone, timedelta

try:
    from zoneinfo import ZoneInfo
except ImportError:
    ZoneInfo = None

# ── 本地时区：记忆库所有时间戳的生成与展示都走这里 ──────────────────
# 默认 Asia/Shanghai（紬 2026-06-23 回国后常驻）；换地方只需设环境变量
# OMBRE_TZ 为 IANA 名（如 Australia/Sydney），无需改代码。
#
# 历史：2026-07 之前这里写死 AEST(UTC+10)，导致她读记忆时间戳会比北京时间
# 快 2 小时，而聊天应用注入的【当前时间】是正确的本地时间——两个时钟打架。
# 旧桶的 created/last_active 是带 +10:00 偏移的完整 ISO，绝对时刻正确，
# 读取时转成本地时区即可，不需要回填。
# 注意 .strip() 必须在 or 之前：否则 OMBRE_TZ=" "（面板里清空变量常留下的空白）
# 是真值，绕过默认值又被 strip 成空串 → 回退 UTC，整库时间戳偏 8 小时。
_TZ_NAME = os.environ.get("OMBRE_TZ", "").strip() or "Asia/Shanghai"
# 容器里没装 tzdata 时 ZoneInfo 会抛错；这几个区全年固定偏移（中国 1991 年后无夏令时），
# 用静态偏移兜底完全等价，不至于整个服务退回 UTC。
_FIXED_FALLBACK = {
    "Asia/Shanghai": 8, "Asia/Hong_Kong": 8, "Asia/Taipei": 8, "Asia/Macau": 8,
    "Asia/Singapore": 8, "PRC": 8, "Asia/Tokyo": 9, "Asia/Seoul": 9,
}


def _load_tz(name):
    if ZoneInfo is not None:
        try:
            return ZoneInfo(name), name
        except Exception:
            pass
    if name in _FIXED_FALLBACK:
        off = _FIXED_FALLBACK[name]
        logging.warning(f"zoneinfo 不可用，{name} 回退为固定 UTC+{off}（该区无夏令时，等价）")
        return timezone(timedelta(hours=off)), name
    logging.warning(f"时区 {name} 不可用且无固定回退，退回 UTC")
    return timezone.utc, "UTC"


LOCAL_TZ, LOCAL_TZ_NAME = _load_tz(_TZ_NAME)


def load_config(config_path: str = None) -> dict:
    """
    Load configuration file.
    加载配置文件。

    Priority: environment variables > config.yaml > built-in defaults.
    优先级：环境变量 > config.yaml > 内置默认值。
    """
    # --- Built-in defaults (fallback so it runs even without config.yaml) ---
    # --- 内置默认配置（兜底，保证即使没有 config.yaml 也能跑）---
    defaults = {
        "transport": "stdio",
        "log_level": "INFO",
        "buckets_dir": os.path.join(os.path.dirname(os.path.abspath(__file__)), "buckets"),
        "merge_threshold": 75,
        "dehydration": {
            "model": "deepseek-chat",
            "base_url": "https://api.deepseek.com/v1",
            "api_key": "",
            # 1024 会把超大桶的摘要 JSON 截断在半截字符串上，导致无法解析
            "max_tokens": 2048,
            "temperature": 0.1,
        },
        "decay": {
            "lambda": 0.05,
            "threshold": 0.3,
            "check_interval_hours": 24,
            "emotion_weights": {
                "base": 1.0,
                "arousal_boost": 0.8,
            },
        },
        "matching": {
            "fuzzy_threshold": 50,
            "max_results": 5,
        },
    }

    # --- Load user config from YAML file ---
    # --- 从 YAML 文件加载用户自定义配置 ---
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "config.yaml"
        )

    config = defaults.copy()
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                file_config = yaml.safe_load(f) or {}
            if isinstance(file_config, dict):
                config = _deep_merge(defaults, file_config)
            else:
                logging.warning(
                    f"Config file is not a valid YAML dict, using defaults / "
                    f"配置文件不是有效的 YAML 字典，使用默认配置: {config_path}"
                )
        except yaml.YAMLError as e:
            logging.warning(
                f"Failed to parse config file, using defaults / "
                f"配置文件解析失败，使用默认配置: {e}"
            )

    # --- Environment variable overrides (highest priority) ---
    # --- 环境变量覆盖敏感/运行时配置（优先级最高）---
    env_api_key = os.environ.get("OMBRE_API_KEY", "")
    if env_api_key:
        config.setdefault("dehydration", {})["api_key"] = env_api_key

    env_base_url = os.environ.get("OMBRE_BASE_URL", "")
    if env_base_url:
        config.setdefault("dehydration", {})["base_url"] = env_base_url

    env_transport = os.environ.get("OMBRE_TRANSPORT", "")
    if env_transport:
        config["transport"] = env_transport

    env_buckets_dir = os.environ.get("OMBRE_BUCKETS_DIR", "")
    if env_buckets_dir:
        config["buckets_dir"] = env_buckets_dir

    # OMBRE_DEHYDRATION_MODEL (with OMBRE_MODEL alias) overrides dehydration.model
    env_dehy_model = os.environ.get("OMBRE_DEHYDRATION_MODEL", "") or os.environ.get("OMBRE_MODEL", "")
    if env_dehy_model:
        config.setdefault("dehydration", {})["model"] = env_dehy_model

    # OMBRE_DEHYDRATION_BASE_URL overrides dehydration.base_url
    env_dehy_base_url = os.environ.get("OMBRE_DEHYDRATION_BASE_URL", "")
    if env_dehy_base_url:
        config.setdefault("dehydration", {})["base_url"] = env_dehy_base_url

    # OMBRE_EMBEDDING_MODEL overrides embedding.model
    env_embed_model = os.environ.get("OMBRE_EMBEDDING_MODEL", "")
    if env_embed_model:
        config.setdefault("embedding", {})["model"] = env_embed_model

    # OMBRE_EMBEDDING_BASE_URL overrides embedding.base_url
    env_embed_base_url = os.environ.get("OMBRE_EMBEDDING_BASE_URL", "")
    if env_embed_base_url:
        config.setdefault("embedding", {})["base_url"] = env_embed_base_url

    # --- Ensure bucket storage directories exist ---
    # --- 确保记忆桶存储目录存在 ---
    buckets_dir = config["buckets_dir"]
    for subdir in ["permanent", "dynamic", "archive"]:
        os.makedirs(os.path.join(buckets_dir, subdir), exist_ok=True)

    return config


def _deep_merge(base: dict, override: dict) -> dict:
    """
    Deep-merge two dicts; override values take precedence.
    深度合并两个字典，override 的值覆盖 base。
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def setup_logging(level: str = "INFO") -> None:
    """
    Initialize logging system.
    初始化日志系统。

    Note: In MCP stdio mode, stdout is occupied by the protocol;
    logs must go to stderr.
    注意：MCP stdio 模式下 stdout 被协议占用，日志只能走 stderr。
    """
    log_level = getattr(logging, level.upper(), None)
    if not isinstance(log_level, int):
        log_level = logging.INFO

    logging.basicConfig(
        level=log_level,
        format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler()],  # StreamHandler defaults to stderr
    )


def generate_bucket_id() -> str:
    """
    Generate a unique bucket ID (12-char short UUID for readability).
    生成唯一的记忆桶 ID（12 位短 UUID，方便人类阅读）。
    """
    return uuid.uuid4().hex[:12]


def strip_wikilinks(text: str) -> str:
    """
    Remove Obsidian wikilink brackets: [[word]] → word
    去除 Obsidian 双链括号
    """
    return re.sub(r"\[\[([^\]]+)\]\]", r"\1", text) if text else text


def sanitize_name(name: str) -> str:
    """
    Sanitize bucket name, keeping only safe characters.
    Prevents path traversal attacks (e.g. ../../etc/passwd).
    清洗桶名称，只保留安全字符。防止路径遍历攻击。
    """
    if not isinstance(name, str):
        return "unnamed"
    cleaned = re.sub(r"[^\w\s\u4e00-\u9fff-]", "", name, flags=re.UNICODE)
    cleaned = cleaned.strip()[:80]
    return cleaned if cleaned else "unnamed"


def safe_path(base_dir: str, filename: str) -> Path:
    """
    Construct a safe file path, ensuring it stays within base_dir.
    Prevents directory traversal.
    构造安全的文件路径，确保最终路径始终在 base_dir 内部。
    """
    base = Path(base_dir).resolve()
    target = (base / filename).resolve()
    if not str(target).startswith(str(base)):
        raise ValueError(
            f"Path safety check failed / 路径安全检查失败: "
            f"{target} is not inside / 不在 {base} 内"
        )
    return target


def count_tokens_approx(text: str) -> int:
    """
    Rough token count estimate.
    粗略估算 token 数。

    Chinese ≈ 1 char = 1.5 tokens, English ≈ 1 word = 1.3 tokens.
    Used to decide whether dehydration is needed; precision not required.
    中文 ≈ 1字=1.5token，英文 ≈ 1词=1.3token。
    用于判断是否需要脱水压缩，不追求精确。
    """
    if not text:
        return 0
    chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
    english_words = len(re.findall(r"[a-zA-Z]+", text))
    return int(chinese_chars * 1.5 + english_words * 1.3 + len(text) * 0.05)


def now_iso() -> str:
    """
    Return current local time as ISO string (with offset, so the instant is unambiguous).
    返回当前本地时间的 ISO 字符串（带时区偏移，绝对时刻无歧义）。
    """
    return datetime.now(LOCAL_TZ).isoformat(timespec="seconds")


def now_local_str(fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Current local time as a naive string (for stores that compare timestamps as text).
    当前本地时间的朴素字符串（用于 events 这类按字符串大小比较的表）。
    """
    return datetime.now(LOCAL_TZ).strftime(fmt)


def parse_ts(value):
    """
    Parse any stored timestamp into an aware datetime; None if unparseable.
    把存储的时间戳解析成带时区的 datetime，失败返回 None。
    兼容三种历史格式：带偏移 ISO（+10:00 旧桶 / +08:00 新桶）、朴素 ISO、
    朴素 "YYYY-MM-DD HH:MM:SS"。朴素值按本地时区解释。
    """
    if not value:
        return None
    s = str(value).strip()
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        dt = None
        for f in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
            try:
                dt = datetime.strptime(s, f)
                break
            except ValueError:
                continue
        if dt is None:
            return None
    return dt if dt.tzinfo else dt.replace(tzinfo=LOCAL_TZ)


def fmt_local(value, fmt: str = "%Y-%m-%d %H:%M") -> str:
    """
    Render a stored timestamp in local time for the model to read.
    把存储的时间戳转成本地时区的可读字符串（解析失败则原样返回）。
    给模型看的时间一律走这里——绝不能再吐带 +10:00 的裸 ISO，
    否则它会把外地时区的钟点当成此刻的本地时间。
    """
    dt = parse_ts(value)
    if dt is None:
        return str(value or "")
    return dt.astimezone(LOCAL_TZ).strftime(fmt)


def ts_sort_key(value):
    """
    Sort key for timestamps. 排序键：跨时区混排时字典序会算错先后，必须比较绝对时刻。
    （旧桶 +10:00 与新桶 +08:00 并存时，"03:57+10:00" 字典序大于 "02:00+08:00"，
    但后者其实更晚——直接 sort 字符串会把新旧顺序排反。）
    """
    dt = parse_ts(value)
    return dt if dt is not None else datetime.min.replace(tzinfo=timezone.utc)


def now_diff_days(value) -> float:
    """
    Days elapsed since a stored timestamp (0 if unparseable-safe caller wants that).
    距某个存储时间戳过去了多少天；无法解析返回 None，由调用方决定兜底值。
    注意：必须用 datetime.now(LOCAL_TZ)（aware）去减 aware 时间戳——
    历史上这里用了朴素的 datetime.now()，与带偏移的时间戳相减必抛 TypeError，
    被 except 吞掉后衰减时间维度形同虚设。
    """
    dt = parse_ts(value)
    if dt is None:
        return None
    return (datetime.now(LOCAL_TZ) - dt).total_seconds() / 86400.0
