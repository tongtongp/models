from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import requests


CURRENT_DIR = Path(__file__).resolve().parent
JWT_FILE = CURRENT_DIR / "09_gen_jwt.py"


spec = importlib.util.spec_from_file_location("qweather_jwt", JWT_FILE)
if spec is None or spec.loader is None:
    raise ImportError(f"无法加载 JWT 文件: {JWT_FILE}")

qweather_jwt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qweather_jwt)

generate_qweather_jwt = qweather_jwt.generate_qweather_jwt


class QWeatherApiError(RuntimeError):
    pass


class QWeatherClient:
    def __init__(
        self,
        api_host: str,
        private_key_path: str | Path = "ed25519-private.pem",
        sub: str = "4FKRV33M9W",
        kid: str = "KJ59BN995H",
        timeout: int = 30,
    ) -> None:
        self.api_host = api_host.rstrip("/")
        self.private_key_path = Path(private_key_path)
        self.sub = sub
        self.kid = kid
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        token = generate_qweather_jwt(
            private_key_path=self.private_key_path,
            sub=self.sub,
            kid=self.kid,
        )
        return {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
        }

    def _get(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.api_host}{path}"
        response = requests.get(
            url,
            headers=self._headers(),
            params=params,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()

        code = data.get("code")
        if code != "200":
            raise QWeatherApiError(
                f"和风天气接口返回异常，code={code}，响应={json.dumps(data, ensure_ascii=False)}"
            )
        return data

    @staticmethod
    def _format_lon_lat(lon: float, lat: float) -> str:
        return f"{lon:.2f},{lat:.2f}"

    def lookup_location_id_by_lonlat(
        self,
        lon: float,
        lat: float,
        adm: str | None = None,
        range_: str | None = None,
        lang: str | None = None,
        number: int = 10,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "location": self._format_lon_lat(lon, lat),
            "number": number,
        }
        if adm:
            params["adm"] = adm
        if range_:
            params["range"] = range_
        if lang:
            params["lang"] = lang

        data = self._get("/geo/v2/city/lookup", params)
        locations = data.get("location", [])
        if not locations:
            raise QWeatherApiError(
                f"未根据经纬度查询到 LocationID，经纬度={lon},{lat}"
            )
        return locations[0]

    def get_historical_weather(
        self,
        location_id: str,
        date: str,
        lang: str | None = None,
        unit: str | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "location": location_id,
            "date": date,
        }
        if lang:
            params["lang"] = lang
        if unit:
            params["unit"] = unit

        return self._get("/v7/historical/weather", params)

    def get_historical_weather_by_lonlat(
        self,
        lon: float,
        lat: float,
        date: str,
        adm: str | None = None,
        range_: str | None = None,
        lang: str | None = None,
        unit: str | None = None,
    ) -> dict[str, Any]:
        location_info = self.lookup_location_id_by_lonlat(
            lon=lon,
            lat=lat,
            adm=adm,
            range_=range_,
            lang=lang,
        )
        history = self.get_historical_weather(
            location_id=location_info["id"],
            date=date,
            lang=lang,
            unit=unit,
        )
        return {
            "location_query": {
                "lon": lon,
                "lat": lat,
                "date": date,
            },
            "location_info": location_info,
            "history": history,
        }


def save_json(data: dict[str, Any], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def read_float_input(prompt: str) -> float:
    value = input(prompt).strip()
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"输入值不是有效数字: {value}") from exc


if __name__ == "__main__":
    API_HOST = "https://nb2k5payfn.re.qweatherapi.com"
    PRIVATE_KEY_PATH = "ed25519-private.pem"
    SUB = "4FKRV33M9W"
    KID = "KJ59BN995H"

    lon = read_float_input("请输入经度（例如 104.07）：")
    lat = read_float_input("请输入纬度（例如 30.54）：")
    date = input("请输入历史日期 yyyyMMdd（例如 20260320）：").strip()

    client = QWeatherClient(
        api_host=API_HOST,
        private_key_path=PRIVATE_KEY_PATH,
        sub=SUB,
        kid=KID,
    )

    result = client.get_historical_weather_by_lonlat(
        lon=lon,
        lat=lat,
        date=date,
        range_="cn",
        lang="zh",
        unit="m",
    )

    save_json(result, CURRENT_DIR / "outputs" / f"qweather_history_{date}.json")
    print(json.dumps(result, ensure_ascii=False, indent=2))