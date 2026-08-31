"""一键验证脚本：覆盖 RBAC/租户隔离/权限缓存/审计/文档 ACL 五个阶段。

用法（先启动服务）:
  .venv\Scripts\python.exe -m uvicorn app.api:app --host 0.0.0.0 --port 8001
  .venv\Scripts\python.exe scripts\verify_all.py

说明:
  - 默认访问 http://localhost:8001，可用环境变量 VERIFY_BASE 覆盖
  - 幂等：每次运行使用唯一后缀，测试数据用后自动清理
  - 任一步骤失败会累计 FAIL，脚本以非 0 退出（便于接入 CI）
  - 依赖: admin/admin123 种子账号 + MySQL 可用（直接入库造测试文档）
"""
import json
import os
import sys
import time
import uuid
import urllib.error
import urllib.request

BASE = os.getenv("VERIFY_BASE", "http://localhost:8001").rstrip("/")
RUN = uuid.uuid4().hex[:8]
PWD = "Verify123!"
ROLE = "verify_{}".format(RUN)
TENANT_A = "vfy_a_{}".format(RUN)
TENANT_B = "vfy_b_{}".format(RUN)
DOC_A = "verify_{}_docA".format(RUN)
DOC_B = "verify_{}_docB".format(RUN)
DOC_C = "verify_{}_docC".format(RUN)

passed = 0
failed = 0


def check(name, ok, detail=""):
    global passed, failed
    print("[{}] {}{}".format("PASS" if ok else "FAIL", name,
                             " - " + detail if detail else ""))
    if ok:
        passed += 1
    else:
        failed += 1


def api(method, path, token=None, payload=None, params=None):
    url = BASE + path
    if params:
        url += "?" + urllib.parse.urlencode(params)
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = "Bearer " + token
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            body = r.read()
            return r.status, json.loads(body.decode()) if body else {}
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode())
        except Exception:
            return e.code, {}
    except Exception as e:
        return -1, {"detail": str(e)}


def login(username, password):
    s, r = api("POST", "/api/auth/login", payload={"username": username, "password": password})
    return (r.get("token"), r) if s == 200 else (None, r)


def _fetch_text(path):
    """抓取纯文本响应（如 /metrics），不做 JSON 解析。"""
    req = urllib.request.Request(BASE + path)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")
    except Exception as e:
        return -1, str(e)


def create_user(admin_token, username, tenant):
    s, r = api("POST", "/api/admin/users", token=admin_token, payload={
        "username": username, "password": PWD, "roles": [ROLE], "tenant_id": tenant,
    })
    if s != 200:
        # 已存在 → 查 id
        _, lu = api("GET", "/api/admin/users", token=admin_token)
        uid = next(x["user_id"] for x in lu["users"] if x["username"] == username)
    else:
        uid = r["user_id"]
    return uid


def doc_list(token):
    s, r = api("GET", "/api/knowledge/documents", token=token)
    return [d["document_id"] for d in r.get("documents", [])] if s == 200 else ["<HTTP %d>" % s]


def seed_doc(doc_id, owner_user_id, tenant):
    from app.storage.document_repository import DocumentRepository
    from app.storage.mysql import MySQLManager
    repo = DocumentRepository(MySQLManager())
    repo.insert(doc_id, doc_id + ".txt", 100, tenant_id=tenant, owner_user_id=owner_user_id)


def cleanup(admin_token):
    """清理本次运行产生的全部数据（尽力而为）。"""
    from app.storage.mysql import MySQLManager
    try:
        mgr = MySQLManager()
        with mgr.get_connection() as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM document_acl WHERE document_id LIKE 'verify_%'")
            cur.execute("DELETE FROM documents WHERE document_id LIKE 'verify_%'")
    except Exception:
        pass
    for u in ("va", "vb", "vc"):
        s, r = api("GET", "/api/admin/users", token=admin_token)
        for x in r.get("users", []):
            if x["username"].startswith("verify_{}_".format(RUN)):
                api("DELETE", "/api/admin/users/" + x["user_id"], token=admin_token)
    api("DELETE", "/api/admin/roles/" + ROLE, token=admin_token)


def main():
    global passed, failed
    print("=" * 72)
    print("一键验证脚本（覆盖 5 个阶段）  BASE={}  RUN={}".format(BASE, RUN))
    print("=" * 72)

    # ---------- 阶段 0：服务健康 ----------
    s, _ = api("GET", "/api/health")
    check("服务健康检查 GET /api/health", s == 200, "HTTP %d" % s)

    # ---------- 阶段 1：认证 / RBAC ----------
    admin_token, r = login("admin", "admin123")
    check("admin 登录（阶段1 认证）", bool(admin_token))
    if not admin_token:
        print("! 无法登录 admin，请确认服务已启动且已运行 scripts/seed_users.py")
        sys.exit(1)

    s, r = api("GET", "/api/auth/me", token=admin_token)
    check("GET /api/auth/me 返回用户信息", s == 200 and r.get("username") == "admin",
          "username=%s perms=%d" % (r.get("username"), len(r.get("permissions", []))))
    check("admin 具备新权限点 knowledge:grant / admin:audit",
          "knowledge:grant" in r.get("permissions", []) and "admin:audit" in r.get("permissions", []))

    # 角色管理（建 → 删）
    s, _ = api("POST", "/api/admin/roles", token=admin_token, payload={
        "role_code": ROLE,
        "name": "验证角色",
        "permissions": ["chat:query", "knowledge:read", "knowledge:upload",
                        "knowledge:delete", "knowledge:rebuild", "knowledge:grant"],
    })
    check("创建角色（阶段1 管理 API）", s == 200, "HTTP %d" % s)
    if s != 200:
        print("! 角色创建失败，后续依赖该角色，退出")
        sys.exit(1)

    # viewer 越权被拒（阶段1 权限门禁）
    vu = "verify_{}_viewer".format(RUN)
    s, _ = api("POST", "/api/admin/users", token=admin_token, payload={
        "username": vu, "password": PWD, "roles": ["viewer"], "tenant_id": "default",
    })
    vt, _ = login(vu, PWD)
    check("viewer 登录", bool(vt))
    s, _ = api("GET", "/api/admin/users", token=vt)
    check("viewer 访问管理 API 被拒（403 权限门禁）", s == 403, "HTTP %d" % s)
    s, _ = api("GET", "/api/knowledge/status", token=vt)
    check("viewer 可读知识库（knowledge:read 放行）", s == 200, "HTTP %d" % s)

    # ---------- 阶段 2 + 5：造数据（两租户三用户三文档）----------
    ua = create_user(admin_token, "verify_{}_va".format(RUN), TENANT_A)  # docA 归属
    ub = create_user(admin_token, "verify_{}_vb".format(RUN), TENANT_A)  # docB 归属
    uc = create_user(admin_token, "verify_{}_vc".format(RUN), TENANT_B)  # docC 归属
    seed_doc(DOC_A, ua, TENANT_A)
    seed_doc(DOC_B, ub, TENANT_A)
    seed_doc(DOC_C, uc, TENANT_B)
    ta, _ = login("verify_{}_va".format(RUN), PWD)
    tb, _ = login("verify_{}_vb".format(RUN), PWD)
    tc, _ = login("verify_{}_vc".format(RUN), PWD)

    # ---------- 阶段 2：租户隔离 ----------
    la, lb, lc = doc_list(ta), doc_list(tb), doc_list(tc)
    check("同租户各自只见自己文档（docA→va / docB→vb）",
          DOC_A in la and DOC_B in lb and DOC_B not in la and DOC_A not in lb,
          "va=%s vb=%s" % (la, lb))
    check("跨租户不可见（vc 看不到租户A 的 docA/docB）",
          DOC_A not in lc and DOC_B not in lc and DOC_C in lc, "vc=%s" % lc)

    # ---------- 阶段 5：文档级 ACL ----------
    s, r = api("DELETE", "/api/knowledge/" + DOC_A, token=tb)
    check("ACL 授权前：vb 删除 docA 被拒（403）",
          s == 403 and "无权删除" in r.get("detail", ""), "HTTP %d %s" % (s, r.get("detail")))

    s, r = api("POST", "/api/knowledge/{}/acl".format(DOC_A), token=ta, payload={
        "principal_type": "user", "principal_id": ub, "permission": "read",
    })
    check("ACL 授权：va(归属人) 授予 vb 对 docA 的 read", s == 200, "HTTP %d" % s)

    lba = doc_list(tb)
    check("ACL 授权后：vb 可见 docA", DOC_A in lba, "vb=%s" % lba)
    s, r = api("DELETE", "/api/knowledge/" + DOC_A, token=tb)
    check("ACL 只授 read：vb 删除 docA 仍 403", s == 403, "HTTP %d %s" % (s, r.get("detail")))
    lc2 = doc_list(tc)
    check("授权不跨租户：vc 仍看不到 docA", DOC_A not in lc2)

    s, r = api("GET", "/api/knowledge/{}/acl".format(DOC_A), token=ta)
    grants = r.get("grants", [])
    check("docA 授权列表含 vb(read)",
          any(g.get("principal_id") == ub and g.get("permission") == "read" for g in grants))

    # 非归属人管理授权被拒
    s, r = api("POST", "/api/knowledge/{}/acl".format(DOC_B), token=ta, payload={
        "principal_type": "user", "principal_id": ua, "permission": "read",
    })
    check("非归属人 va 不能给 docB 授权（403）", s == 403, "HTTP %d %s" % (s, r.get("detail")))

    # ---------- 阶段 4：审计日志 ----------
    time.sleep(0.6)  # 等待审计 worker 落库
    s, r = api("GET", "/api/admin/audit-logs", token=admin_token)
    actions = [x.get("action") for x in r.get("logs", [])]
    check("审计日志可查询且含 authz.denied（阶段4）",
          s == 200 and "authz.denied" in actions, "total=%d actions=%s" % (len(actions), set(actions)))

    # ---------- 阶段 3：权限感知缓存指标 ----------
    s, text = _fetch_text("/metrics")
    check("缓存指标暴露（阶段3）",
          s == 200 and "rag_query_cache_hits_total" in text and "rag_query_cache_misses_total" in text,
          "HTTP %d" % s)

    # ---------- 清理 ----------
    cleanup(admin_token)
    print("\n" + "=" * 72)
    print("验证结果: {} PASS / {} FAIL".format(passed, failed))
    print("=" * 72)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    import urllib.parse
    main()
