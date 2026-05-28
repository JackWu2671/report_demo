"""
独立 SQL 查询测试脚本 — 无任何项目依赖，直接运行即可。

用法：
    python services_test/test_sql_query.py

修改下方 CONFIG 和 QUERY 两个区域的参数后运行。
"""

import json
import logging
import time
import uuid

import requests
import urllib3

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ── 填写真实值 ────────────────────────────────────────────────────────────────

CONFIG = {
    "base_url":  "",   # 例: "https://<HOST>:<PORT>/rest/nmeatomapigatewayservice/open-api/v1"
    "operator":  "",   # 例: "10110"
    "country":   "",   # 例: "10000044"
    "user_id":   "",   # 例: "test15"
    "app_name":  "smart-u",
    "domain":    "",   # 例: "数通"
    "data_type": "AGGR",
    "verify_ssl": False,
    "timeout":    60,
    "max_retries": 20,  # 轮询最大次数
    "poll_interval": 0.5,  # 每次轮询间隔（秒）
}

QUERY = {
    "sql":        "SELECT neType FROM ads_aggr_unb_eval_ip_networkelement LIMIT 5",
    "table_name": "ads_aggr_unb_eval_ip_networkelement",
}

# ─────────────────────────────────────────────────────────────────────────────


def submit_task(session: requests.Session, cfg: dict, sql: str, table_name: str) -> str | None:
    url = f"{cfg['base_url']}/api-task"
    body = {
        "apiId":    "SQL_EXECUTOR",
        "operator": cfg["operator"],
        "country":  cfg["country"],
        "userId":   cfg["user_id"],
        "param": {
            "cache": "true",
            "param": json.dumps({
                "country":   cfg["country"],
                "dataId":    None,
                "appName":   cfg["app_name"],
                "domain":    cfg["domain"],
                "dataType":  cfg["data_type"],
                "userId":    cfg["user_id"],
                "operator":  cfg["operator"],
                "sql":       sql,
                "tableName": table_name,
            }, ensure_ascii=False),
            "operationId": str(uuid.uuid4()),
            "type":    "ATOM",
            "nodeId":  str(uuid.uuid4()),
            "version": "LATEST",
            "apiId":   "SQL_EXECUTOR",
        },
    }

    log.info("POST %s", url)
    try:
        resp = session.post(url, json=body, verify=cfg["verify_ssl"], timeout=cfg["timeout"])
    except requests.exceptions.RequestException as e:
        log.error("POST 失败: %s", e)
        return None

    if resp.status_code != 200:
        log.error("HTTP %d: %s", resp.status_code, resp.text[:300])
        return None

    data = resp.json()
    if not data.get("status"):
        log.error("提交失败: %s", data.get("msg"))
        return None

    task_id = data.get("data")
    log.info("任务 ID: %s", task_id)
    return task_id


def poll_result(session: requests.Session, cfg: dict, task_id: str) -> list | None:
    url = f"{cfg['base_url']}/api-task?taskId={task_id}"
    for i in range(cfg["max_retries"]):
        time.sleep(cfg["poll_interval"])
        try:
            resp = session.get(url, verify=cfg["verify_ssl"], timeout=cfg["timeout"])
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            log.error("轮询 %d/%d 异常: %s", i + 1, cfg["max_retries"], e)
            continue

        if data.get("status") is True and data.get("data") is not None:
            try:
                outer = json.loads(data["data"])
                inner = json.loads(outer.get("data", "{}"))
                rows  = inner.get("datas")
                log.info("获取到 %d 行", len(rows) if rows else 0)
                return rows
            except Exception as e:
                log.error("结果解析失败: %s | raw=%s", e, str(data)[:200])
                return None

        log.info("轮询 %d/%d: 处理中...", i + 1, cfg["max_retries"])

    log.warning("轮询超时（%d 次）", cfg["max_retries"])
    return None


def main():
    cfg = CONFIG
    sql        = QUERY["sql"]
    table_name = QUERY["table_name"]

    if not cfg["base_url"]:
        log.error("请先填写 CONFIG['base_url']")
        return

    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    with requests.Session() as session:
        session.headers.update({"Content-Type": "application/json"})

        task_id = submit_task(session, cfg, sql, table_name)
        if not task_id:
            return

        rows = poll_result(session, cfg, task_id)

    if rows:
        log.info("查询成功，共 %d 行:", len(rows))
        for i, row in enumerate(rows):
            log.info("  [%d] %s", i, row)
    else:
        log.warning("查询无结果")


if __name__ == "__main__":
    main()
