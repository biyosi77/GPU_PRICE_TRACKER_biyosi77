#!/usr/bin/env python3
"""
Daily AI GPU Rental Price Tracker
- Collects hourly rental prices (USD/GPU/hr) and available GPU counts from:
  1. Vast.ai Public API (console.vast.ai/api/v0/bundles/)
  2. RunPod Public GraphQL API (api.runpod.io/graphql)
  3. Price of Compute Public API (priceofcompute.com/api/v1/prices/:sku & /history/:sku?format=csv)
- Normalizes metrics, calculates DoD (1D), WoW (7D), and MoM (30D) delta ($) & % changes
- Appends/upserts to data/daily_gpu_prices.csv and data/daily_gpu_prices.json
- Generates minimalist dark-mode 30-day daily trend chart (reports/gpu_daily_trends.png)
- Generates Markdown briefing report in reports/
- Sends Telegram briefing if TELEGRAM_BOT_TOKEN & TELEGRAM_CHAT_ID are set
"""

import argparse
import io
import json
import os
import statistics
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import requests

KST = timezone(timedelta(hours=9))
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
REPORTS_DIR = BASE_DIR / "reports"
CSV_PATH = DATA_DIR / "daily_gpu_prices.csv"
JSON_PATH = DATA_DIR / "daily_gpu_prices.json"
CHART_PATH = REPORTS_DIR / "gpu_daily_trends.png"

TARGET_MODELS = [
    "H100 SXM",
    "H100 PCIe",
    "H200",
    "B200",
    "B300",
    "A100 80GB",
]

POC_SLUGS = {
    "H100 SXM": "h100-sxm",
    "H100 PCIe": "h100-pcie",
    "H200": "h200-sxm",
    "B200": "b200",
    "B300": "b300",
    "A100 80GB": "a100-sxm-80gb",
}

HEADERS = {
    "User-Agent": "gpu-rental-price-tracker/1.0 (+https://github.com)",
    "Accept": "application/json, text/csv;q=0.9, */*;q=0.8",
}


def normalize_vast_model(gpu_name: str, gpu_ram_mb: float = 0) -> Optional[str]:
    name = (gpu_name or "").upper().strip()
    if "B300" in name:
        return "B300"
    if "B200" in name:
        return "B200"
    if "H200" in name:
        return "H200"
    if "H100" in name:
        if "PCIE" in name or "NVL" in name:
            return "H100 PCIe"
        return "H100 SXM"
    if "A100" in name:
        if gpu_ram_mb and gpu_ram_mb < 60000:
            return None
        if "80" in name or "SXM" in name or gpu_ram_mb >= 60000:
            return "A100 80GB"
    return None


def normalize_runpod_model(gpu_id: str, display_name: str, mem_gb: float = 0) -> Optional[str]:
    text = f"{gpu_id} {display_name}".upper()
    if "B300" in text:
        return "B300"
    if "B200" in text:
        return "B200"
    if "H200" in text:
        return "H200"
    if "H100" in text:
        if "PCIE" in text or "NVL" in text:
            return "H100 PCIe"
        return "H100 SXM"
    if "A100" in text and (mem_gb >= 75 or "80" in text):
        return "A100 80GB"
    return None


def fetch_vastai_data(timeout: int = 20) -> Dict[str, Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {
        m: {"ondemand_prices": [], "spot_prices": [], "available_gpus": 0}
        for m in TARGET_MODELS
    }
    query = {
        "rentable": {"eq": True},
        "verified": {"eq": True},
        "order": [["dph_total", "asc"]],
        "limit": 1000,
    }
    url = "https://console.vast.ai/api/v0/bundles/"
    resp = requests.get(url, params={"q": json.dumps(query)}, headers=HEADERS, timeout=timeout)
    resp.raise_for_status()
    offers = resp.json().get("offers", [])

    for offer in offers:
        model = normalize_vast_model(offer.get("gpu_name", ""), float(offer.get("gpu_ram") or 0))
        if not model:
            continue
        num_gpus = int(offer.get("num_gpus") or 1)
        if num_gpus <= 0:
            continue
        dph_total = offer.get("dph_total")
        min_bid = offer.get("min_bid")

        results[model]["available_gpus"] += num_gpus
        if dph_total and float(dph_total) > 0:
            results[model]["ondemand_prices"].append(float(dph_total) / num_gpus)
        if min_bid and float(min_bid) > 0:
            results[model]["spot_prices"].append(float(min_bid) / num_gpus)

    return results


def fetch_runpod_data(timeout: int = 15) -> Dict[str, Dict[str, Any]]:
    url = "https://api.runpod.io/graphql"
    query_full = """
    query {
      gpuTypes {
        id
        displayName
        memoryInGb
        securePrice
        communityPrice
        lowestPrice(input: { gpuCount: 1 }) {
          minimumBidPrice
          uninterruptablePrice
          stockStatus
        }
      }
    }
    """
    query_simple = """
    query {
      gpuTypes {
        id
        displayName
        memoryInGb
        securePrice
        communityPrice
      }
    }
    """
    resp = requests.post(url, json={"query": query_full}, headers=HEADERS, timeout=timeout)
    if resp.status_code != 200 or "errors" in resp.json():
        resp = requests.post(url, json={"query": query_simple}, headers=HEADERS, timeout=timeout)
    resp.raise_for_status()

    out: Dict[str, Dict[str, Any]] = {}
    for g in resp.json().get("data", {}).get("gpuTypes", []):
        model = normalize_runpod_model(
            g.get("id", ""),
            g.get("displayName", ""),
            float(g.get("memoryInGb") or 0),
        )
        if not model:
            continue
        lp = g.get("lowestPrice") or {}
        out[model] = {
            "secure_price": g.get("securePrice"),
            "community_price": g.get("communityPrice"),
            "spot_min_bid": lp.get("minimumBidPrice"),
            "stock_status": lp.get("stockStatus"),
        }
    return out


def _fetch_poc_history_series(slug: str, timeout: int = 15) -> Dict[str, float]:
    """Fetch full daily on-demand median history from Price of Compute (CSV first, JSON fallback)."""
    series: Dict[str, float] = {}

    # 1) Try ?format=csv first (cleanest flat table)
    try:
        csv_resp = requests.get(
            f"https://priceofcompute.com/api/v1/history/{slug}",
            params={"format": "csv"},
            headers=HEADERS,
            timeout=timeout,
        )
        if csv_resp.status_code == 200 and "," in csv_resp.text:
            hdf = pd.read_csv(io.StringIO(csv_resp.text))
            cols_lower = {c.lower().strip(): c for c in hdf.columns}
            date_col = next((cols_lower[k] for k in ("day", "date", "timestamp") if k in cols_lower), None)
            if date_col:
                if "pricing_type" in cols_lower:
                    pt_col = cols_lower["pricing_type"]
                    od_rows = hdf[hdf[pt_col].astype(str).str.lower() == "on_demand"]
                    val_col = next(
                        (cols_lower[k] for k in ("usd_per_gpu_hr", "median", "price", "value") if k in cols_lower),
                        None,
                    )
                    if val_col:
                        for _, r in od_rows.iterrows():
                            if pd.notna(r[date_col]) and pd.notna(r[val_col]):
                                series[str(r[date_col])[:10]] = float(r[val_col])
                else:
                    val_col = next(
                        (
                            cols_lower[k]
                            for k in ("on_demand", "on_demand_median", "ondemand_median_usd", "usd_per_gpu_hr", "median", "price")
                            if k in cols_lower
                        ),
                        None,
                    )
                    if val_col:
                        for _, r in hdf.iterrows():
                            if pd.notna(r[date_col]) and pd.notna(r[val_col]):
                                series[str(r[date_col])[:10]] = float(r[val_col])
            if series:
                return series
    except Exception:
        pass

    # 2) Fallback to JSON endpoint with flexible key matching
    try:
        j_resp = requests.get(
            f"https://priceofcompute.com/api/v1/history/{slug}",
            headers=HEADERS,
            timeout=timeout,
        )
        if j_resp.status_code == 200:
            payload = j_resp.json()
            rows = []
            if isinstance(payload, list):
                rows = payload
            elif isinstance(payload, dict):
                for key in ("history", "data", "series", "points", "rows", "items"):
                    if isinstance(payload.get(key), list):
                        rows = payload[key]
                        break
                # Handle dict-of-lists e.g. {"on_demand": [{"day": ..., "usd_per_gpu_hr": ...}]}
                if not rows and isinstance(payload.get("series"), dict):
                    od_list = payload["series"].get("on_demand") or []
                    if isinstance(od_list, list):
                        rows = od_list
                if not rows and isinstance(payload.get("on_demand"), list):
                    rows = payload["on_demand"]

            for item in rows:
                if not isinstance(item, dict):
                    continue
                day = item.get("day") or item.get("date") or item.get("timestamp")
                val = (
                    item.get("on_demand")
                    or item.get("ondemand_median_usd")
                    or item.get("usd_per_gpu_hr")
                    or item.get("median")
                    or item.get("price")
                    or item.get("value")
                )
                if isinstance(val, dict):
                    val = val.get("usd_per_gpu_hr") or val.get("median") or val.get("price")
                if day and val is not None:
                    series[str(day)[:10]] = float(val)
    except Exception:
        pass

    return series


def fetch_priceofcompute_data(timeout: int = 15) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for model, slug in POC_SLUGS.items():
        try:
            resp = requests.get(
                f"https://priceofcompute.com/api/v1/prices/{slug}",
                headers=HEADERS,
                timeout=timeout,
            )
            if resp.status_code != 200:
                continue
            payload = resp.json()
            prices = payload.get("prices", {})
            providers = payload.get("providers", [])
            od_list = [
                float(p["usd_per_gpu_hr"])
                for p in providers
                if p.get("pricing_type") == "on_demand" and p.get("usd_per_gpu_hr")
            ]
            spot_list = [
                float(p["usd_per_gpu_hr"])
                for p in providers
                if p.get("pricing_type") in ("spot", "interruptible", "community")
                and p.get("usd_per_gpu_hr")
            ]

            history_series = _fetch_poc_history_series(slug, timeout=timeout)
            print(f"[PriceOfCompute] {model}: {len(history_series)} daily history points loaded")

            out[model] = {
                "od_median": (prices.get("on_demand") or {}).get("usd_per_gpu_hr"),
                "od_providers": (prices.get("on_demand") or {}).get("providers", len(od_list)),
                "spot_median": (prices.get("spot") or prices.get("community") or {}).get("usd_per_gpu_hr"),
                "od_list": od_list,
                "spot_list": spot_list,
                "history": history_series,
            }
        except Exception as exc:
            print(f"[PriceOfCompute] {model} warning: {exc}")
            continue
    return out


def _calc_delta(cur_val: Optional[float], prev_val: Optional[float]) -> Tuple[Optional[float], Optional[float]]:
    if cur_val is None or prev_val is None or pd.isna(cur_val) or pd.isna(prev_val) or float(prev_val) <= 0:
        return None, None
    diff = round(float(cur_val) - float(prev_val), 2)
    pct = round((diff / float(prev_val)) * 100, 2)
    return diff, pct


def aggregate_daily_records(today_str: str) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, float]]]:
    vast_data: Dict[str, Dict[str, Any]] = {}
    runpod_data: Dict[str, Dict[str, Any]] = {}
    poc_data: Dict[str, Dict[str, Any]] = {}

    try:
        vast_data = fetch_vastai_data()
        print(f"[Vast.ai] fetched {sum(v['available_gpus'] for v in vast_data.values())} GPUs")
    except Exception as exc:
        print(f"[Vast.ai] fetch warning: {exc}")

    try:
        runpod_data = fetch_runpod_data()
        print(f"[RunPod] fetched {len(runpod_data)} target models")
    except Exception as exc:
        print(f"[RunPod] fetch warning: {exc}")

    try:
        poc_data = fetch_priceofcompute_data()
    except Exception as exc:
        print(f"[PriceOfCompute] fetch warning: {exc}")

    records: List[Dict[str, Any]] = []
    external_history: Dict[str, Dict[str, float]] = {}

    for model in TARGET_MODELS:
        v = vast_data.get(model, {})
        rp = runpod_data.get(model, {})
        poc = poc_data.get(model, {})
        if poc.get("history"):
            external_history[model] = poc["history"]

        od_pool: List[float] = []
        od_pool.extend(v.get("ondemand_prices", []))
        od_pool.extend(poc.get("od_list", []))
        if rp.get("secure_price"):
            od_pool.append(float(rp["secure_price"]))
        if rp.get("community_price"):
            od_pool.append(float(rp["community_price"]))

        spot_pool: List[float] = []
        spot_pool.extend(v.get("spot_prices", []))
        spot_pool.extend(poc.get("spot_list", []))
        if rp.get("spot_min_bid"):
            spot_pool.append(float(rp["spot_min_bid"]))

        od_min = round(min(od_pool), 2) if od_pool else None
        od_median = (
            round(float(poc["od_median"]), 2)
            if poc.get("od_median")
            else (round(statistics.median(od_pool), 2) if od_pool else None)
        )
        od_mean = round(statistics.mean(od_pool), 2) if od_pool else None

        spot_min = round(min(spot_pool), 2) if spot_pool else None
        spot_median = (
            round(float(poc["spot_median"]), 2)
            if poc.get("spot_median")
            else (round(statistics.median(spot_pool), 2) if spot_pool else None)
        )

        records.append(
            {
                "date": today_str,
                "gpu_model": model,
                "ondemand_min_usd": od_min,
                "ondemand_median_usd": od_median,
                "ondemand_mean_usd": od_mean,
                "spot_min_usd": spot_min,
                "spot_median_usd": spot_median,
                "vast_available_gpus": int(v.get("available_gpus") or 0),
                "runpod_secure_usd": round(float(rp["secure_price"]), 2) if rp.get("secure_price") else None,
                "runpod_stock_status": rp.get("stock_status") or "N/A",
                "poc_providers": int(poc.get("od_providers") or 0),
            }
        )
    return records, external_history


def _lookup_external_ref(ext_hist: Dict[str, float], target_date_str: str) -> Optional[float]:
    if not ext_hist:
        return None
    candidates = [d for d in sorted(ext_hist.keys()) if d <= target_date_str]
    if not candidates:
        return None
    return ext_hist[candidates[-1]]


def upsert_timeseries(
    new_records: List[Dict[str, Any]],
    external_history: Optional[Dict[str, Dict[str, float]]] = None,
    backfill_days: int = 30,
) -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    external_history = external_history or {}

    # Seed past 30 days of historical daily points if available from external_history
    backfill_rows: List[Dict[str, Any]] = []
    if external_history and backfill_days > 0:
        today_dt = datetime.now(KST).date()
        cutoff_str = (today_dt - timedelta(days=backfill_days)).strftime("%Y-%m-%d")
        for model, hist_map in external_history.items():
            for d_str, val in hist_map.items():
                if d_str >= cutoff_str and d_str < today_dt.strftime("%Y-%m-%d"):
                    backfill_rows.append(
                        {
                            "date": d_str,
                            "gpu_model": model,
                            "ondemand_median_usd": round(float(val), 2),
                        }
                    )

    frames = []
    if CSV_PATH.exists():
        frames.append(pd.read_csv(CSV_PATH))
    if backfill_rows:
        frames.append(pd.DataFrame(backfill_rows))
    frames.append(pd.DataFrame(new_records))

    combined = pd.concat(frames, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"]).dt.strftime("%Y-%m-%d")
    # Prefer existing full rows or today's new_records over sparse backfill rows
    combined = combined.drop_duplicates(subset=["date", "gpu_model"], keep="last")
    combined = combined.sort_values(["date", "gpu_model"]).reset_index(drop=True)

    delta_cols = {
        "dod_change_usd": [],
        "dod_change_pct": [],
        "wow_7d_change_usd": [],
        "wow_7d_change_pct": [],
        "mom_30d_change_usd": [],
        "mom_30d_change_pct": [],
    }

    for _, row in combined.iterrows():
        cur_date = pd.to_datetime(row["date"])
        model = row["gpu_model"]
        cur_val = row["ondemand_median_usd"]
        ext_hist = external_history.get(model, {})

        model_hist = combined[
            (combined["gpu_model"] == model) & (pd.to_datetime(combined["date"]) < cur_date)
        ]

        d1_usd, d1_pct = (row.get("dod_change_usd"), row.get("dod_change_pct"))
        if pd.isna(d1_usd) or pd.isna(d1_pct):
            prev_1d = model_hist.iloc[-1]["ondemand_median_usd"] if not model_hist.empty else _lookup_external_ref(
                ext_hist, (cur_date - timedelta(days=1)).strftime("%Y-%m-%d")
            )
            d1_usd, d1_pct = _calc_delta(cur_val, prev_1d)

        d7_usd, d7_pct = (row.get("wow_7d_change_usd"), row.get("wow_7d_change_pct"))
        if pd.isna(d7_usd) or pd.isna(d7_pct):
            target_7d = (cur_date - timedelta(days=7)).strftime("%Y-%m-%d")
            hist_7d = model_hist[model_hist["date"] <= target_7d]
            prev_7d = hist_7d.iloc[-1]["ondemand_median_usd"] if not hist_7d.empty else _lookup_external_ref(
                ext_hist, target_7d
            )
            d7_usd, d7_pct = _calc_delta(cur_val, prev_7d)

        d30_usd, d30_pct = (row.get("mom_30d_change_usd"), row.get("mom_30d_change_pct"))
        if pd.isna(d30_usd) or pd.isna(d30_pct):
            target_30d = (cur_date - timedelta(days=30)).strftime("%Y-%m-%d")
            hist_30d = model_hist[model_hist["date"] <= target_30d]
            prev_30d = hist_30d.iloc[-1]["ondemand_median_usd"] if not hist_30d.empty else _lookup_external_ref(
                ext_hist, target_30d
            )
            d30_usd, d30_pct = _calc_delta(cur_val, prev_30d)

        delta_cols["dod_change_usd"].append(d1_usd if pd.notna(d1_usd) else None)
        delta_cols["dod_change_pct"].append(d1_pct if pd.notna(d1_pct) else None)
        delta_cols["wow_7d_change_usd"].append(d7_usd if pd.notna(d7_usd) else None)
        delta_cols["wow_7d_change_pct"].append(d7_pct if pd.notna(d7_pct) else None)
        delta_cols["mom_30d_change_usd"].append(d30_usd if pd.notna(d30_usd) else None)
        delta_cols["mom_30d_change_pct"].append(d30_pct if pd.notna(d30_pct) else None)

    for col, vals in delta_cols.items():
        combined[col] = vals

    combined.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
    combined.to_json(JSON_PATH, orient="records", indent=2, force_ascii=False)
    return combined


def generate_daily_charts(
    df: pd.DataFrame,
    external_history: Optional[Dict[str, Dict[str, float]]] = None,
    days: int = 30,
) -> Path:
    """Render minimalist dark-mode daily area/line charts (Silicon Data style)."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    external_history = external_history or {}

    bg_outer = "#0D1017"
    bg_panel = "#12161F"
    border_col = "#222834"
    grid_col = "#242B38"
    text_main = "#F1F5F9"
    text_muted = "#94A3B8"
    line_col = "#64D2FF"

    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5), dpi=180)
    fig.patch.set_facecolor(bg_outer)
    axes_flat = axes.flatten()

    cmap = LinearSegmentedColormap.from_list("cyan_fade", [(0.39, 0.82, 1.0, 0.0), (0.39, 0.82, 1.0, 0.24)])

    for idx, model in enumerate(TARGET_MODELS):
        ax = axes_flat[idx]
        ax.set_facecolor(bg_panel)
        for spine in ax.spines.values():
            spine.set_color(border_col)
            spine.set_linewidth(0.9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        series_map: Dict[str, float] = {}
        for d_str, val in external_history.get(model, {}).items():
            series_map[d_str] = float(val)

        sub = df[df["gpu_model"] == model].dropna(subset=["ondemand_median_usd"])
        for _, r in sub.iterrows():
            series_map[str(r["date"])[:10]] = float(r["ondemand_median_usd"])

        if series_map:
            dates_sorted = sorted(series_map.keys())[-days:]
            x_dt = pd.to_datetime(dates_sorted)
            y_vals = np.array([series_map[d] for d in dates_sorted], dtype=float)

            ax.plot(x_dt, y_vals, color=line_col, linewidth=1.8, zorder=3)
            y_min, y_max = float(y_vals.min()), float(y_vals.max())
            pad = max((y_max - y_min) * 0.25, 0.25)
            y_bot = max(0.0, y_min - pad)
            y_top = y_max + pad

            z = np.linspace(0, 1, 100).reshape(-1, 1)
            x_num = mdates.date2num(x_dt)
            if len(x_num) > 1:
                im = ax.imshow(
                    z,
                    aspect="auto",
                    cmap=cmap,
                    origin="lower",
                    extent=[x_num[0], x_num[-1], y_bot, y_top],
                    zorder=1,
                )
                poly = ax.fill_between(x_dt, y_bot, y_vals, facecolor="none", edgecolor="none", zorder=2)
                im.set_clip_path(poly.get_paths()[0], transform=ax.transData)
            else:
                ax.scatter(x_dt, y_vals, color=line_col, s=28, zorder=4)

            ax.set_ylim(y_bot, y_top)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
            if len(x_dt) >= 12:
                ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
            else:
                ax.set_xticks(x_dt)
            ax.set_title(f"{model}  (${y_vals[-1]:.2f}/hr)", loc="left", color=text_main, fontsize=11, fontweight="bold", pad=8)
        else:
            ax.set_title(f"{model}  (No data)", loc="left", color=text_muted, fontsize=11, pad=8)

        ax.yaxis.set_major_formatter("${x:.2f}")
        ax.tick_params(colors=text_muted, labelsize=8.5)
        ax.grid(True, linestyle="--", alpha=0.55, color=grid_col, linewidth=0.7)

    fig.suptitle(
        "AI GPU Daily Rental Price Trend (30D On-Demand Median, USD/GPU/hr)",
        color=text_main,
        fontsize=14,
        fontweight="bold",
        x=0.03,
        ha="left",
        y=0.97,
    )
    fig.text(
        0.03,
        0.015,
        "Source: Vast.ai, RunPod, Price of Compute · Auto-updated daily",
        color=text_muted,
        fontsize=8.5,
    )
    plt.tight_layout(rect=[0.01, 0.035, 0.99, 0.94])
    plt.savefig(CHART_PATH, dpi=180, facecolor=bg_outer)
    plt.close(fig)
    print(f"[Chart] Saved daily trend chart to {CHART_PATH}")
    return CHART_PATH


def fmt_usd(val: Any) -> str:
    return f"${float(val):.2f}" if pd.notna(val) and val is not None else "—"


def fmt_delta(usd_val: Any, pct_val: Any) -> str:
    if (pd.isna(usd_val) or usd_val is None) and (pd.isna(pct_val) or pct_val is None):
        return "—"
    parts = []
    if pd.notna(usd_val) and usd_val is not None:
        u = float(usd_val)
        u_sign = "+" if u > 0 else ("-" if u < 0 else "")
        parts.append(f"{u_sign}${abs(u):.2f}")
    if pd.notna(pct_val) and pct_val is not None:
        p = float(pct_val)
        p_sign = "+" if p > 0 else ""
        parts.append(f"({p_sign}{p:.1f}%)")
    return " ".join(parts)


def generate_markdown_report(df: pd.DataFrame, target_date: str) -> str:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    today_df = df[df["date"] == target_date].copy()

    order_map = {m: i for i, m in enumerate(TARGET_MODELS)}
    today_df["_order"] = today_df["gpu_model"].map(lambda x: order_map.get(x, 999))
    today_df = today_df.sort_values("_order")

    lines = [
        f"# AI GPU 일일 임대가격 & 시계열 변동폭 브리핑 ({target_date} KST)",
        "",
        "단위: USD / GPU / hour · 소스: Vast.ai, RunPod, Price of Compute",
        "",
        "| GPU 모델 | 온디맨드 최저 | 온디맨드 중간값 | 스팟 최저 | 스팟 중간값 | 전일 대비 (1D) | 1주일 대비 (7D) | 한달 대비 (30D) | Vast.ai 가용량 | RunPod 정가 |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    total_vast = 0
    for _, r in today_df.iterrows():
        avail = int(r["vast_available_gpus"]) if pd.notna(r["vast_available_gpus"]) else 0
        total_vast += avail
        lines.append(
            f"| **{r['gpu_model']}** "
            f"| {fmt_usd(r['ondemand_min_usd'])} "
            f"| {fmt_usd(r['ondemand_median_usd'])} "
            f"| {fmt_usd(r['spot_min_usd'])} "
            f"| {fmt_usd(r['spot_median_usd'])} "
            f"| {fmt_delta(r.get('dod_change_usd'), r.get('dod_change_pct'))} "
            f"| {fmt_delta(r.get('wow_7d_change_usd'), r.get('wow_7d_change_pct'))} "
            f"| {fmt_delta(r.get('mom_30d_change_usd'), r.get('mom_30d_change_pct'))} "
            f"| {avail:,} GPUs "
            f"| {fmt_usd(r.get('runpod_secure_usd'))} |"
        )

    lines.extend(
        [
            "",
            f"- **총 오픈 마켓플레이스(Vast.ai) 가용 물량:** {total_vast:,} GPUs",
            f"- **생성 시각:** {datetime.now(KST).strftime('%Y-%m-%d %H:%M:%S KST')}",
        ]
    )

    report_text = "\n".join(lines) + "\n"
    (REPORTS_DIR / f"daily_report_{target_date}.md").write_text(report_text, encoding="utf-8")
    (REPORTS_DIR / "latest_report.md").write_text(report_text, encoding="utf-8")
    return report_text


def send_telegram_briefing(df: pd.DataFrame, target_date: str) -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("[Telegram] TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set; skipping.")
        return

    today_df = df[df["date"] == target_date]
    order_map = {m: i for i, m in enumerate(TARGET_MODELS)}
    today_df = today_df.assign(_order=today_df["gpu_model"].map(lambda x: order_map.get(x, 999))).sort_values("_order")

    msg_lines = [f"📊 <b>AI GPU 일일 임대가격 브리핑 ({target_date})</b>\n"]
    for _, r in today_df.iterrows():
        avail = int(r["vast_available_gpus"]) if pd.notna(r["vast_available_gpus"]) else 0
        msg_lines.append(
            f"• <b>{r['gpu_model']}</b>: 중간값 {fmt_usd(r['ondemand_median_usd'])} "
            f"(최저 {fmt_usd(r['ondemand_min_usd'])} | 스팟 {fmt_usd(r['spot_median_usd'])})\n"
            f"  └ 1D: {fmt_delta(r.get('dod_change_usd'), r.get('dod_change_pct'))} | "
            f"7D: {fmt_delta(r.get('wow_7d_change_usd'), r.get('wow_7d_change_pct'))} | "
            f"30D: {fmt_delta(r.get('mom_30d_change_usd'), r.get('mom_30d_change_pct'))} · 가용 {avail:,}대"
        )
    text = "\n".join(msg_lines)
    resp = requests.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
        timeout=15,
    )
    resp.raise_for_status()
    resp.raise_for_status()
    if CHART_PATH.exists():
        with open(CHART_PATH, "rb") as img_f:
            p_resp = requests.post(
                f"https://api.telegram.org/bot{token}/sendPhoto",
                data={"chat_id": chat_id, "caption": f"📈 AI GPU 30일 임대가격 추이 ({target_date})"},
                files={"photo": img_f},
                timeout=30,
            )
            p_resp.raise_for_status()
    print("[Telegram] Briefing sent successfully.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Daily AI GPU Rental Price Tracker")
    parser.add_argument(
        "--seed-file",
        type=str,
        default=None,
        help="Optional local JSON file to test normalization/report generation offline.",
    )
    args = parser.parse_args()

    today_str = datetime.now(KST).strftime("%Y-%m-%d")
    ext_hist: Dict[str, Dict[str, float]] = {}
    if args.seed_file:
        seed_records = json.loads(Path(args.seed_file).read_text(encoding="utf-8"))
        target_date = max(r["date"] for r in seed_records)
        df = upsert_timeseries(seed_records)
    else:
        target_date = today_str
        records, ext_hist = aggregate_daily_records(today_str)
        df = upsert_timeseries(records, ext_hist, backfill_days=30)

    generate_daily_charts(df, ext_hist)
    report = generate_markdown_report(df, target_date)
    print(report)
    send_telegram_briefing(df, target_date)


if __name__ == "__main__":
    main()
