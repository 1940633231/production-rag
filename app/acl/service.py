"""文档级 ACL 决策服务：集中承载可复用的授权判定逻辑。

从 app/api/knowledge.py 抽取，供路由层与潜在的其他调用方复用。
所有授权判定一律 **fail-closed**（安全视角）：
  - 可读文档集合：非 superadmin 查询异常时返回空集（不放行任何文档）
  - 删除 / 管理权限：判定异常时一律拒绝（deny），宁可误拒不越权
"""
from app.core.logger import get_logger

logger = get_logger(__name__)


def readable_document_ids(user, tenant_id: str):
    """计算用户可读文档集合；None = 不设文档级过滤（鉴权关闭 / superadmin）。

    fail-closed：非 superadmin 用户 ACL 查询异常时返回空集，
    宁可查无结果（列表为空 / 检索无结果）也不放行全部文档造成泄露。
    """
    if user is None or user.is_superadmin:
        return None
    try:
        from app.acl.repository import ACLRepository
        return ACLRepository().get_readable_document_ids(user, tenant_id)
    except Exception as e:
        logger.error(
            "ACL 可读文档计算失败，fail-closed（空集，不放行任何文档）: %s",
            e, exc_info=True,
        )
        return set()


def can_delete(user, tenant_id: str, document_id: str) -> bool:
    """判断用户是否有权删除文档（superadmin / owner / delete 授权）。

    fail-closed：ACL 判定异常时一律拒绝（deny），宁可误拒也不越权放行删除。
    """
    if user is None or user.is_superadmin:
        return True
    try:
        from app.acl.repository import ACLRepository
        return ACLRepository().has_permission(user, document_id, "delete", tenant_id)
    except Exception as e:
        logger.error(
            "ACL 删除权限判定失败，fail-closed（拒绝删除）: %s", e, exc_info=True,
        )
        return False


def can_manage_acl(user, tenant_id: str, document_id: str) -> bool:
    """判断用户是否有权管理文档授权（superadmin / 文档归属人）。"""
    if user is None or user.is_superadmin:
        return True
    try:
        from app.storage.document_repository import DocumentRepository
        doc = DocumentRepository().get(document_id, tenant_id=tenant_id)
        return doc is not None and doc.get("owner_user_id") == user.user_id
    except Exception as e:
        logger.warning("ACL 管理权限判定失败: %s", e)
        return False