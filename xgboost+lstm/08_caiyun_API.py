from __future__ import annotations

from pathlib import Path
import importlib.util
from typing import Any
import time
import uuid
import hmac
import hashlib
import requests
import base64
import urllib.parse


# ===== 动态加载 01_config.py =====
_cfg_spec = importlib.util.spec_from_file_location(
    "cfg", Path(__file__).with_name("01_config.py")
)
cfg = importlib.util.module_from_spec(_cfg_spec)
assert _cfg_spec and _cfg_spec.loader
_cfg_spec.loader.exec_module(cfg)


def to_float(value: Any, default: float = 0.0) -> float:
    """
    将输入值安全转成 float。
    """
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if text == "":
        return default
    try:
        return float(text)
    except ValueError:
        return default


def validate_lat_lon_days(lat: float, lon: float, days: int) -> None:
    """
    校验经纬度与预报天数。
    """
    if not (-90.0 <= lat <= 90.0):
        raise ValueError(f"纬度超出范围: {lat}")
    if not (-180.0 <= lon <= 180.0):
        raise ValueError(f"经度超出范围: {lon}")
    if not (1 <= days <= 15):
        raise ValueError(f"days 必须在 1 到 15 之间，当前传入: {days}")


def generate_caiyun_signature(
    app_key: str,
    app_secret: str,
    method: str,
    path: str,
    query_params: dict[str, str],
    timestamp: str,
    nonce: str,
) -> str:
    """
    按彩云 v2.6 文档生成 x-cy-signature。
    文档要求：
    1. query 参数按字母顺序排序
    2. URL 编码
    3. 按 {method}:{path}:{query}:{app_key}:{nonce}:{timestamp} 拼接
    4. HMAC-SHA256
    5. URL-safe Base64 编码
    """

    sorted_keys = sorted(query_params.keys())

    query_parts = []
    for k in sorted_keys:
        encoded_key = urllib.parse.quote_plus(str(k), safe="")
        encoded_value = urllib.parse.quote_plus(str(query_params[k]), safe="")
        query_parts.append(f"{encoded_key}={encoded_value}")
    query_string = "&".join(query_parts)

    string_to_sign = ":".join([
        method,
        path,
        query_string,
        app_key,
        nonce,
        timestamp,
    ])

    h = hmac.new(
        app_secret.encode("utf-8"),
        string_to_sign.encode("utf-8"),
        hashlib.sha256,
    )

    signature = base64.urlsafe_b64encode(h.digest()).decode("utf-8")
    return signature


def fetch_caiyun_daily_forecast(
    lat: float,
    lon: float,
    days: int,
    app_key: str,
    app_secret: str,
    timeout: int = 20,
) -> dict[str, Any]:
    """
    根据经纬度调用彩云天气 v2.6 天级别预报 API，返回完整 JSON。
    """
    validate_lat_lon_days(lat, lon, days)

    if not app_key:
        raise ValueError("app_key 不能为空")
    if not app_secret:
        raise ValueError("app_secret 不能为空")

    base_url = "https://api.caiyunapp.com"
    path = f"/v2.6/{app_key}/{lon:.4f},{lat:.4f}/daily"
    query_params = {
        "dailysteps": str(days)
    }

    query_string = urllib.parse.urlencode(query_params)
    url = f"{base_url}{path}?{query_string}"

    method = "GET"
    timestamp = str(int(time.time()))
    nonce = str(uuid.uuid4())

    signature = generate_caiyun_signature(
        app_key=app_key,
        app_secret=app_secret,
        method=method,
        path=path,
        query_params=query_params,
        timestamp=timestamp,
        nonce=nonce,
    )

    headers = {
        "x-cy-nonce": nonce,
        "x-cy-timestamp": timestamp,
        "x-cy-signature": signature,
        "Accept": "application/json",
    }

    response = requests.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()

    data = response.json()

    if data.get("status") != "ok":
        raise RuntimeError(f"彩云 API 返回异常: {data}")

    return data


def normalize_caiyun_daily_response(
    api_json: dict[str, Any],
    last_history_row: dict[str, Any],
) -> list[dict[str, float | str]]:
    """
    将彩云 daily 响应标准化为模型可识别的未来天气日记录列表。
    """
    result = api_json.get("result", {})
    daily = result.get("daily", {})

    temperature_list = daily.get("temperature", [])
    precipitation_list = daily.get("precipitation", [])
    wind_list = daily.get("wind", [])
    humidity_list = daily.get("humidity", [])
    pressure_list = daily.get("pressure", [])
    radiation_list = daily.get("dswrf", [])

    n = min(
        len(temperature_list),
        len(precipitation_list),
        len(wind_list),
        len(humidity_list),
        len(pressure_list),
        len(radiation_list),
    )

    if n == 0:
        raise RuntimeError(f"彩云 daily 响应缺少有效日预报字段: {api_json}")

    normalized_rows = []

    for i in range(n):
        temp_item = temperature_list[i]
        precip_item = precipitation_list[i]
        wind_item = wind_list[i]
        humidity_item = humidity_list[i]
        pressure_item = pressure_list[i]
        radiation_item = radiation_list[i]

        date = str(temp_item.get("date", ""))[:10]

        temp_max_c = to_float(temp_item.get("max"))
        temp_min_c = to_float(temp_item.get("min"))
        temp_avg_c = to_float(temp_item.get("avg"))

        precip_max = to_float(precip_item.get("max"))
        precip_min = to_float(precip_item.get("min"))
        precip_sum = to_float(precip_item.get("avg"))

        wind_max = to_float(wind_item.get("max", {}).get("speed"))
        wind_min = to_float(wind_item.get("min", {}).get("speed"))
        wind_avg = to_float(wind_item.get("avg", {}).get("speed"))

        # 彩云 humidity 是 0-1，需要转为百分数
        relative_humidity_max = to_float(humidity_item.get("max")) * 100.0
        relative_humidity_min = to_float(humidity_item.get("min")) * 100.0
        relative_humidity = to_float(humidity_item.get("avg")) * 100.0

        pressure_kpa = to_float(pressure_item.get("avg"))
        pressure_max_kpa = to_float(pressure_item.get("max"))
        pressure_min_kpa = to_float(pressure_item.get("min"))

        radiation_avg = to_float(radiation_item.get("avg"))
        radiation_max = to_float(radiation_item.get("max"))
        radiation_min = to_float(radiation_item.get("min"))

        # 彩云 daily 文档里没有直接给下面这些字段，先用最近历史值补
        # pressure_kpa = to_float(last_history_row.get("pressure_kpa"))
        # pressure_max_kpa = to_float(last_history_row.get("pressure_max_kpa"), pressure_kpa)
        # pressure_min_kpa = to_float(last_history_row.get("pressure_min_kpa"), pressure_kpa)

        soil_moisture = to_float(last_history_row.get("soil_moisture"))
        surface_temp_avg_c = to_float(last_history_row.get("surface_temp_avg_c"), temp_avg_c)
        surface_temp_max_c = to_float(last_history_row.get("surface_temp_max_c"), temp_max_c)
        surface_temp_min_c = to_float(last_history_row.get("surface_temp_min_c"), temp_min_c)

        # radiation_avg = to_float(last_history_row.get("radiation_avg"))
        # radiation_max = to_float(last_history_row.get("radiation_max"), radiation_avg)
        # radiation_min = to_float(last_history_row.get("radiation_min"), radiation_avg)

        soil_rel_humidity = to_float(last_history_row.get("soil_rel_humidity"))
        soil_temp_c = to_float(last_history_row.get("soil_temp_c"))

        row = {
            "date": date,

            "wind_avg": wind_avg,
            "wind_max": wind_max,
            "wind_min": wind_min,

            "precip_max": precip_max,
            "precip_min": precip_min,
            "precip_sum": precip_sum,

            "relative_humidity": relative_humidity,
            "relative_humidity_max": relative_humidity_max,
            "relative_humidity_min": relative_humidity_min,

            "temp_avg_c": temp_avg_c,
            "temp_max_c": temp_max_c,
            "temp_min_c": temp_min_c,

            "pressure_kpa": pressure_kpa,
            "pressure_max_kpa": pressure_max_kpa,
            "pressure_min_kpa": pressure_min_kpa,

            "radiation_avg": radiation_avg,
            "radiation_max": radiation_max,
            "radiation_min": radiation_min,

            "soil_moisture": soil_moisture,

            "surface_temp_avg_c": surface_temp_avg_c,
            "surface_temp_max_c": surface_temp_max_c,
            "surface_temp_min_c": surface_temp_min_c,

            "soil_rel_humidity": soil_rel_humidity,
            "soil_temp_c": soil_temp_c,
        }

        normalized_rows.append(row)

    return normalized_rows


def get_forecast_by_latlon(
    lat: float,
    lon: float,
    days: int,
    last_history_row: dict[str, Any],
    app_key: str,
    app_secret: str,
) -> list[dict[str, float | str]]:
    """
    给后续界面层调用的高层函数：
    输入经纬度 + 预报天数，返回标准化后的未来天气记录。
    """
    api_json = fetch_caiyun_daily_forecast(
        lat=lat,
        lon=lon,
        days=days,
        app_key=app_key,
        app_secret=app_secret,
    )

    normalized_rows = normalize_caiyun_daily_response(
        api_json=api_json,
        last_history_row=last_history_row,
    )
    return normalized_rows


if __name__ == "__main__":
    # ===== 本地标准化测试：构造一个模拟彩云响应 =====
    sample_api_json = {
        "status": "ok",
        "result": {
            "daily": {
                "temperature": [
                    {"date": "2026-03-25T00:00+08:00", "max": 29, "min": 18, "avg": 23.5},
                    {"date": "2026-03-26T00:00+08:00", "max": 30, "min": 19, "avg": 24.5},
                ],
                "precipitation": [
                    {"date": "2026-03-25T00:00+08:00", "max": 3.2, "min": 0.0, "avg": 3.2},
                    {"date": "2026-03-26T00:00+08:00", "max": 0.0, "min": 0.0, "avg": 0.0},
                ],
                "wind": [
                    {
                        "date": "2026-03-25T00:00+08:00",
                        "max": {"speed": 12.0},
                        "min": {"speed": 6.0},
                        "avg": {"speed": 9.0},
                    },
                    {
                        "date": "2026-03-26T00:00+08:00",
                        "max": {"speed": 10.0},
                        "min": {"speed": 5.0},
                        "avg": {"speed": 7.5},
                    },
                ],
                "humidity": [
                    {"date": "2026-03-25T00:00+08:00", "max": 0.88, "min": 0.80, "avg": 0.84},
                    {"date": "2026-03-26T00:00+08:00", "max": 0.91, "min": 0.85, "avg": 0.88},
                ],
            }
        }
    }

    sample_last_history_row = {
        "soil_moisture": 26.5,
        "surface_temp_avg_c": 23.0,
        "surface_temp_max_c": 31.0,
        "surface_temp_min_c": 17.5,
        "pressure_kpa": 91.2,
        "pressure_max_kpa": 91.5,
        "pressure_min_kpa": 90.8,
        "radiation_avg": 145.0,
        "radiation_max": 210.0,
        "radiation_min": 80.0,
        "soil_rel_humidity": 72.0,
        "soil_temp_c": 21.4,
    }

    normalized = normalize_caiyun_daily_response(
        api_json=sample_api_json,
        last_history_row=sample_last_history_row,
    )

    print("标准化后的未来天气记录：")
    for row in normalized:
        print(row)

    # ===== 真实 API 请求测试 =====
    RUN_REAL_API_TEST = True

    if RUN_REAL_API_TEST:
        APP_KEY = "j3n8gh3kqs8iqz6i"
        APP_SECRET = "TfGQf7fDKHmiaNP8RiEb2ddeTTV3cYD4"

        lat = float(input("请输入纬度，例如 29.67：").strip())
        lon = float(input("请输入经度，例如 102.23：").strip())
        forecast_days = int(input("请输入预报天数（1-15，建议 3 或 7）：").strip())

        api_json = fetch_caiyun_daily_forecast(
            lat=lat,
            lon=lon,
            days=forecast_days,
            app_key=APP_KEY,
            app_secret=APP_SECRET,
        )

        print("\n彩云 API 原始返回：")
        print(api_json)

        normalized_real = normalize_caiyun_daily_response(
            api_json=api_json,
            last_history_row=sample_last_history_row,
        )

        print("\n标准化后的真实预报结果：")
        for row in normalized_real:
            print(row)