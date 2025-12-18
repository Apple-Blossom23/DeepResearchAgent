import psycopg2
from typing import List, Dict, Any, Optional
from db_pool_manager import db_pool_manager, initialize_db_pool
import json

def _ensure_db_pool():
    status = db_pool_manager.get_pool_status()
    if not status.get("initialized"):
        initialize_db_pool()

def init_results_table() -> None:
    _ensure_db_pool()
    with db_pool_manager.get_cursor(commit=True) as cursor:
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS workflow_answers (
                id SERIAL PRIMARY KEY,
                task_id TEXT NOT NULL,
                workflow_category TEXT,
                answer_type TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

def insert_answers(task_id: str, items: List[Dict[str, Any]]) -> None:
    if not items:
        return
    _ensure_db_pool()
    with db_pool_manager.get_cursor(commit=True) as cursor:
        for it in items:
            cursor.execute(
                """
                INSERT INTO workflow_answers (task_id, workflow_category, answer_type, content)
                VALUES (%s, %s, %s, %s)
                """,
                (
                    task_id,
                    it.get("workflow_category"),
                    it.get("answer_type"),
                    it.get("content"),
                ),
            )

def query_answers(task_id: Optional[str] = None, category: Optional[str] = None, answer_type: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
    _ensure_db_pool()
    sql = "SELECT id, task_id, workflow_category, answer_type, content, created_at FROM workflow_answers"
    clauses = []
    params: List[Any] = []
    if task_id:
        clauses.append("task_id = %s")
        params.append(task_id)
    if category:
        clauses.append("workflow_category = %s")
        params.append(category)
    if answer_type:
        clauses.append("answer_type = %s")
        params.append(answer_type)
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY created_at DESC LIMIT %s"
    params.append(limit)
    with db_pool_manager.get_cursor(commit=False) as cursor:
        cursor.execute(sql, tuple(params))
        rows = cursor.fetchall()
    return [
        {
            "id": r[0],
            "task_id": r[1],
            "workflow_category": r[2],
            "answer_type": r[3],
            "content": r[4],
            "created_at": r[5].isoformat() if hasattr(r[5], "isoformat") else str(r[5]),
        }
        for r in rows
    ]

def aggregate_parallel_results(parallel_results: Dict[str, Any]) -> str:
    try:
        items = []
        for cat, r in parallel_results.items():
            status = getattr(r, 'status', None) or (r.get('status') if isinstance(r, dict) else None)
            exec_time = getattr(r, 'execution_time', None) or (r.get('execution_time') if isinstance(r, dict) else None)
            reasoning = getattr(r, 'reasoning', None) or (r.get('reasoning') if isinstance(r, dict) else [])
            snippet = "\n".join([str(x) for x in reasoning]) if reasoning else ""
            items.append({
                "category": cat,
                "status": status,
                "execution_time": exec_time,
                "reasoning": snippet,
            })
        summary = {
            "categories": [it["category"] for it in items],
            "completed": sum(1 for it in items if it.get("status") == "completed"),
            "failed": sum(1 for it in items if it.get("status") == "failed"),
            "timeout": sum(1 for it in items if it.get("status") == "timeout"),
        }
        payload = {"summary": summary, "details": items}
        return json.dumps(payload, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"summary": {"error": str(e)}, "details": []}, ensure_ascii=False)
