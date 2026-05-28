# 标准库
import json
import logging
import os
import time
import uuid
from typing import Any, Dict, List, Optional

# 第三方库
import requests
import urllib3
import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    加载配置文件。

    默认路径：backend/config.yaml
    """
    if config_path is None:
        # 从 services/ 向上两级找到 backend/
        _services_dir = os.path.dirname(os.path.abspath(__file__))
        _backend_dir  = os.path.dirname(_services_dir)
        config_path   = os.path.join(_backend_dir, "config.yaml")

    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"配置文件不存在: {config_path}\n"
            f"请复制 config.example.yaml 为 config.yaml 并填写配置"
        )

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class DeApiClient:
    """SQL 查询 API 客户端（异步任务式：POST 提交 → GET 轮询结果）"""

    # ── 初始化 ────────────────────────────────────────────────
    def __init__(self,
                 config_path: str = None,
                 base_url: str = None,
                 city: str = None,
                 operator: str = None,
                 province: str = None,
                 user_id: str = None,
                 country: str = None,
                 app_name: str = None,
                 domain: str = None,
                 data_type: str = None,
                 verify_ssl: bool = None,
                 timeout: int = None,
                 max_retries: int = None):
        explicit_params = {
            "base_url":    base_url,
            "city":        city,
            "operator":    operator,
            "province":    province,
            "user_id":     user_id,
            "country":     country,
            "app_name":    app_name,
            "domain":      domain,
            "data_type":   data_type,
            "verify_ssl":  verify_ssl,
            "timeout":     timeout,
            "max_retries": max_retries,
        }
        api_config = self._load_api_config(config_path)
        self._apply_config(explicit_params, api_config)
        self._validate_config()
        self.session: requests.Session = None

    def __enter__(self):
        self._init_session()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            self.session.close()

    # ── 公共接口 ──────────────────────────────────────────────
    def execute_sql_query(self, sql: str, table_name: str) -> Optional[List[Dict]]:
        """
        执行 SQL 查询，返回结果列表。失败返回 None。

        Args:
            sql        : SQL 语句
            table_name : 主表名（用于任务标识）
        """
        task_id = self._submit_task(sql, table_name)
        if not task_id:
            return None
        return self._get_task_result(task_id)

    def execute_sql_query_with_retry(self,
                                     sql: str,
                                     table_name: str,
                                     retry_count: int = 3) -> Optional[List[Dict]]:
        """带重试的 SQL 查询（每次间隔 3 秒）"""
        for attempt in range(retry_count):
            if attempt > 0:
                logger.info("第 %d 次重试...", attempt + 1)
                time.sleep(3)
            result = self.execute_sql_query(sql, table_name)
            if result is not None:
                return result
        logger.warning("执行失败，已重试 %d 次", retry_count)
        return None

    # ── 内部方法 ──────────────────────────────────────────────
    @staticmethod
    def _get_config_defaults() -> Dict[str, Any]:
        return {
            "base_url":    "",
            "city":        "",
            "operator":    "",
            "province":    "",
            "user_id":     "",
            "country":     "",
            "app_name":    "",
            "domain":      "",
            "data_type":   "AGGR",
            "verify_ssl":  False,
            "timeout":     60,
            "max_retries": 10,
        }

    @staticmethod
    def _load_api_config(config_path: str) -> Dict[str, Any]:
        try:
            return load_config(config_path).get("api", {})
        except FileNotFoundError:
            return {}

    @staticmethod
    def _resolve_config_value(explicit, from_config, default):
        if explicit is not None:
            return explicit
        if from_config is not None:
            return from_config
        return default

    def _apply_config(self, explicit: Dict, api_config: Dict) -> None:
        for key, default in self._get_config_defaults().items():
            value = self._resolve_config_value(
                explicit.get(key), api_config.get(key), default
            )
            setattr(self, key, value)

    def _validate_config(self) -> None:
        if not self.base_url:
            raise ValueError("base_url 未配置，请检查 config.yaml")

    def _init_session(self) -> None:
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
        if not self.verify_ssl:
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    def _submit_task(self, sql: str, table_name: str) -> Optional[str]:
        post_url  = f"{self.base_url}/api-task"
        post_data = self._build_task_request(sql, table_name)
        logger.info("提交查询: %s", table_name)
        response  = self._do_post_request(post_url, post_data)
        if response is None:
            return None
        return self._extract_task_id(response)

    def _build_task_request(self, sql: str, table_name: str) -> Dict[str, Any]:
        return {
            "apiId":    "SQL_EXECUTOR",
            "operator": self.operator,
            "country":  self.country,
            "userId":   self.user_id,
            "param": {
                "cache": "true",
                "param": json.dumps({
                    "country":   self.country,
                    "dataId":    None,
                    "appName":   self.app_name,
                    "domain":    self.domain,
                    "dataType":  self.data_type,
                    "userId":    self.user_id,
                    "operator":  self.operator,
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

    def _do_post_request(self, url: str, data: Dict) -> Optional[requests.Response]:
        try:
            resp = self.session.post(url, json=data,
                                     verify=self.verify_ssl, timeout=self.timeout)
            if resp.status_code != 200:
                logger.error("POST 失败 %d: %s", resp.status_code, resp.text[:200])
                return None
            return resp
        except requests.exceptions.RequestException as e:
            logger.error("POST 请求异常: %s", e)
            return None

    @staticmethod
    def _extract_task_id(response: requests.Response) -> Optional[str]:
        try:
            result = response.json()
            if not result.get("status"):
                logger.error("提交失败: %s", result.get("msg"))
                return None
            task_id = result.get("data")
            logger.info("任务 ID: %s", task_id)
            return task_id
        except json.JSONDecodeError as e:
            logger.error("响应解析失败: %s", e)
            return None

    def _get_task_result(self, task_id: str) -> Optional[List[Dict]]:
        get_url = f"{self.base_url}/api-task?taskId={task_id}"
        logger.info("轮询结果...")
        for i in range(self.max_retries):
            time.sleep(0.5)
            result = self._fetch_once(get_url, i)
            if result is not None:
                return result
        logger.warning("轮询超时（%d 次）", self.max_retries)
        return None

    def _fetch_once(self, url: str, attempt: int) -> Optional[List[Dict]]:
        try:
            resp = self.session.get(url, verify=self.verify_ssl, timeout=self.timeout)
            resp.raise_for_status()
            result = resp.json()
            if result.get("status") is True and result.get("data") is not None:
                return self._parse_nested_data(result["data"])
            logger.debug("轮询 %d/%d: 处理中...", attempt + 1, self.max_retries)
            return None
        except (requests.exceptions.RequestException, json.JSONDecodeError, KeyError) as e:
            logger.error("轮询异常: %s", e)
            return None

    @staticmethod
    def _parse_nested_data(data_str: str) -> Optional[List[Dict]]:
        parsed   = json.loads(data_str)
        inner    = json.loads(parsed.get("data", "{}"))
        datas    = inner.get("datas")
        logger.info("获取到 %d 条数据", len(datas) if datas else 0)
        return datas
