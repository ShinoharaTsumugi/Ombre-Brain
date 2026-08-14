# ============================================================
# MCP transport auth wrapper / MCP 通道鉴权包装
#
# Why this file exists / 为什么有这个文件：
#   server.py exposes the native MCP endpoint (/mcp) via FastMCP's
#   streamable_http_app() with ONLY a CORS middleware — no auth. So anyone
#   who knows the URL can call breath/hold/grow/trace and read or modify all
#   memories. The dashboard (_require_auth) and the REST mirror
#   (/api/*, _check_api_auth) are already protected; only /mcp was left open.
#
#   This wrapper closes that hole WITHOUT editing server.py, so the fork can
#   still merge upstream (P0luz/Ombre-Brain) cleanly — it only adds a new file.
#   It reuses the existing OMBRE_API_TOKEN (the same token the Telegram bot and
#   the REST endpoints already use), so all clients share one credential.
#
# How to run (set as the start command, e.g. in Zeabur service settings):
#   uvicorn mcp_auth:app --host 0.0.0.0 --port 8000
# ============================================================

import hmac
import os

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse

# Importing server runs its module-level setup (config, engines, MCP tools,
# dashboard/REST routes) — same as launching it, minus the __main__ block.
from server import mcp, _verify_any_password

OMBRE_API_TOKEN = os.environ.get("OMBRE_API_TOKEN", "").strip()

# 公开秘密路径（给官方 Claude Chat 连接器用）：把 /mcp 额外挂在一个带秘密前缀的路径上，
# 该路径不鉴权、直接放行（中间件改写回 /mcp）。Claude 连这个 URL 就像连一个开放 MCP，
# 不触发 OAuth。安全性 = 这串 secret 的保密性（和 token 等价），勿外泄。
# 在 Zeabur 服务环境变量里设 MCP_PUBLIC_PATH_SECRET=<一长串随机>，连接器 URL 即：
#   https://<你的域名>/<MCP_PUBLIC_PATH_SECRET>/mcp
# 留空则关闭此功能（只剩 token 鉴权的 /mcp，app/bot 照常用）。
MCP_PUBLIC_SECRET = os.environ.get("MCP_PUBLIC_PATH_SECRET", "").strip()

# Paths that carry the raw MCP transport and must require a token.
# Everything else (/, /health, /auth/*, /dashboard, /api/*) keeps its own
# auth (or is intentionally public) and is left untouched.
_PROTECTED_PREFIXES = ("/mcp", "/sse")


def _token_ok(token: str) -> bool:
    """Mirror server._check_api_auth: accept OMBRE_API_TOKEN or the dashboard password."""
    if not token:
        return False
    if OMBRE_API_TOKEN and hmac.compare_digest(token, OMBRE_API_TOKEN):
        return True
    try:
        return _verify_any_password(token)
    except Exception:
        return False


class MCPAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        path = request.url.path
        # 公开秘密路径：/<secret>/mcp[...] → 改写成 /mcp[...] 并免鉴权放行（Claude Chat 连接器）。
        # 仅当 path 精确带正确 secret 前缀时生效；无 secret 的人访问普通 /mcp 仍需 token。
        if MCP_PUBLIC_SECRET:
            prefix = "/" + MCP_PUBLIC_SECRET
            if path == prefix + "/mcp" or path.startswith(prefix + "/mcp/"):
                new_path = path[len(prefix):]
                request.scope["path"] = new_path
                request.scope["raw_path"] = new_path.encode("utf-8")
                return await call_next(request)
        protected = any(
            path == p or path.startswith(p + "/") for p in _PROTECTED_PREFIXES
        )
        # Let CORS preflight through; the CORS middleware (outer) answers it.
        if protected and request.method != "OPTIONS":
            auth = request.headers.get("authorization", "")
            if auth[:7].lower() == "bearer ":
                token = auth[7:].strip()
            else:
                token = request.headers.get("x-api-token", "").strip()
            if not _token_ok(token):
                return JSONResponse(
                    {"error": "Unauthorized: MCP endpoint requires a valid token"},
                    status_code=401,
                )
        return await call_next(request)


# Build the same ASGI app server.py would, then layer auth under CORS.
# add_middleware wraps last-added outermost, so CORS stays outermost (handles
# preflight + adds headers even to 401s) and auth runs just inside it.
app = mcp.streamable_http_app()
app.add_middleware(MCPAuthMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


if __name__ == "__main__":
    # 直接作为容器入口（Dockerfile: CMD ["python", "mcp_auth.py"]）：
    # = server.py 的全部能力 + /mcp Bearer 鉴权门 + 连接器秘密路径。
    # 2026-08-14 教训：迁移到 Git 构建后 CMD 曾是裸 server.py，中间件没跑，
    # /mcp 在公网裸奔了一天——鉴权入口必须就是默认入口，不靠平台自定义启动命令。
    import uvicorn

    try:
        port = int(os.environ.get("OMBRE_PORT", "8000") or "8000")
    except ValueError:
        port = 8000
    uvicorn.run(app, host="0.0.0.0", port=port)
