"""Mission food mass from a typical day and an EVA day.

:func:`mission_food` asks :func:`optimizer.optimize_menu` for two minimum-mass
menus. The typical day sets every crew member's EVA time to 0. The EVA day
uses that member's own ``eva_hours`` when the key is present, and otherwise
``eva_hours`` from the call (default 6.5). Mission length is split by an
EVA-day fraction, then each food's servings are scaled by ``1 + safety_margin``.

EVA-day fraction, when the caller does not pass one:

    clip(ISS EVA count / span of those EVA dates in days, 0, 1)

The count and the span come from ``data/launch_library/spacewalks.json``
(a Launch Library 2 cache). See :data:`EVA_FRACTION_FORMULA`.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from data_loader import DATA_DIR
from optimizer import optimize_menu

# ISS EVA count / span of those EVA dates in days, clipped to [0, 1].
# Span is the latest ISS EVA start minus the earliest, in days (86400 seconds).
# Results are filtered to spacestation.name == "International Space Station"
# when that field exists. Undated events are left out. A zero span with at
# least one dated ISS EVA yields 1 (every day in that span is an EVA day).
# No dated ISS EVA yields 0.
EVA_FRACTION_FORMULA = (
    "ISS EVA count divided by the span of those EVA dates in days, clipped to [0, 1]. "
    "Span is the latest ISS EVA start minus the earliest, in days (86400 seconds). "
    "Launch Library results are filtered to spacestation.name == "
    "'International Space Station' when that field exists. Undated events are left out. "
    "A zero span with at least one dated ISS EVA yields 1; no dated ISS EVA yields 0."
)

ISS_STATION_NAME = "International Space Station"
_DAY_SECONDS = 86400.0
_DATE_KEYS = ("start", "date", "end")


def mission_food(
    days: int | float,
    crew: Sequence[Mapping] | Mapping,
    *,
    safety_margin: float = 0.10,
    eva_day_fraction: float | None = None,
    eva_hours: float = 6.5,
    foods: pd.DataFrame | None = None,
    allergies: Sequence[str] | str | None = None,
    data_dir: Path | str | None = None,
) -> dict:
    """Pack food for ``days`` of this crew, with a safety margin on servings.

    The typical day and the EVA day are each a minimum-mass menu from
    :func:`optimizer.optimize_menu`. ``crew`` and ``allergies`` use that
    function's shapes. On the typical day every member's ``eva_hours`` is 0.
    On the EVA day a member keeps their own ``eva_hours`` when that key is
    already set; otherwise the call's ``eva_hours`` (default 6.5) is used.

    ``eva_day_fraction`` is the share of mission days that include an EVA,
    from 0 to 1 inclusive. Pass it directly, or leave it unset to read
    ``launch_library/spacewalks.json`` under ``data_dir`` (the package
    ``data`` directory by default). The file formula is
    :data:`EVA_FRACTION_FORMULA`, and the same text is returned in
    ``eva_day_fraction_source``. A missing file raises ``FileNotFoundError``
    and names ``eva_day_fraction``; there is no silent default.

    ``typical_days = days * (1 - fraction)`` and
    ``eva_days = days * fraction``. For each food that appears in either
    menu::

        servings = (typical_servings * typical_days + eva_servings * eva_days)
                   * (1 + safety_margin)
        mass_kg = servings * mass_g_per_serving / 1000

    Grams per serving are the optimizer's packaged mass contribution divided
    by that day's servings (``mass_g / servings``), which is the foods table
    ``mass_g``. Foods whose scaled servings are 0 are omitted. The packing
    list is sorted by ``mass_kg`` descending.

    If either day is infeasible, ``packing_list`` is empty, ``total_mass_kg``
    is ``None``, and ``message`` names each failing day (``typical`` or
    ``EVA``) and the unmet nutrients. No menu is invented for that mission.

    Returns:
        Dict with ``feasible``, ``days``, ``typical_days``, ``eva_days``,
        ``eva_day_fraction``, ``eva_day_fraction_source``, ``safety_margin``,
        ``packing_list`` (``item``, ``servings``, ``mass_kg``),
        ``total_mass_kg``, ``typical_day``, ``eva_day``, and ``message``.
        The day values are the optimizer results (``feasible``,
        ``total_mass_g``, ``unmet``, and the rest of that dict).

    Raises:
        ValueError: ``days`` is not positive, ``safety_margin`` is negative,
            or ``eva_day_fraction`` is outside 0 to 1.
        FileNotFoundError: ``eva_day_fraction`` was omitted and the
            spacewalks cache is not in the data directory.
    """

    mission_days = _positive_days(days)
    margin = _non_negative_margin(safety_margin)
    fraction, source = _resolve_fraction(eva_day_fraction, data_dir)
    typical_days = mission_days * (1.0 - fraction)
    eva_days = mission_days * fraction

    typical = optimize_menu(
        _crew_for_day(crew, eva=False, eva_hours=eva_hours),
        foods=foods,
        allergies=allergies,
        data_dir=data_dir,
    )
    eva = optimize_menu(
        _crew_for_day(crew, eva=True, eva_hours=eva_hours),
        foods=foods,
        allergies=allergies,
        data_dir=data_dir,
    )

    base = {
        "days": mission_days,
        "typical_days": typical_days,
        "eva_days": eva_days,
        "eva_day_fraction": fraction,
        "eva_day_fraction_source": source,
        "safety_margin": margin,
        "typical_day": typical,
        "eva_day": eva,
    }
    if not typical["feasible"] or not eva["feasible"]:
        return {
            **base,
            "feasible": False,
            "packing_list": [],
            "total_mass_kg": None,
            "message": _infeasible_message(typical, eva),
        }

    packing_list = _packing_list(typical, eva, typical_days, eva_days, margin)
    total_mass_kg = float(sum(row["mass_kg"] for row in packing_list))
    return {
        **base,
        "feasible": True,
        "packing_list": packing_list,
        "total_mass_kg": total_mass_kg,
        "message": (
            f"Mission food is {total_mass_kg:.3f} kg for {mission_days:g} days "
            f"({typical_days:g} typical, {eva_days:g} EVA) "
            f"with a {margin:.0%} safety margin."
        ),
    }


def eva_fraction_from_spacewalks(path: Path | str | None = None) -> float:
    """Return the ISS EVA-day fraction in ``[0, 1]`` from a spacewalks cache.

    ``path`` defaults to ``data/launch_library/spacewalks.json``. The value
    follows :data:`EVA_FRACTION_FORMULA`.

    Raises:
        FileNotFoundError: The cache is missing. Pass ``eva_day_fraction``
            to :func:`mission_food`.
        ValueError: The file is not a usable Launch Library payload.
    """

    fraction, _source = _fraction_from_file(_spacewalks_file(path))
    return fraction


def _resolve_fraction(
    eva_day_fraction: float | None,
    data_dir: Path | str | None,
) -> tuple[float, str]:
    if eva_day_fraction is not None:
        fraction = _unit_interval(eva_day_fraction)
        source = f"caller: eva_day_fraction={fraction:g}. Formula not applied ({EVA_FRACTION_FORMULA})"
        return fraction, source
    path = spacewalks_path(data_dir)
    return _fraction_from_file(path)


def spacewalks_path(data_dir: Path | str | None = None) -> Path:
    """Path to ``launch_library/spacewalks.json`` under the food data directory."""

    root = DATA_DIR if data_dir is None else Path(data_dir)
    return root / "launch_library" / "spacewalks.json"


def _spacewalks_file(path: Path | str | None) -> Path:
    if path is None:
        return spacewalks_path(None)
    return Path(path)


def _fraction_from_file(path: Path) -> tuple[float, str]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Launch Library spacewalks file not found: {path}. "
            "Pass eva_day_fraction (0 to 1 inclusive) to mission_food."
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{path} is not valid JSON. Pass eva_day_fraction (0 to 1 inclusive) to mission_food."
        ) from exc
    results = _results_list(payload, path)
    selected = _iss_events(results)
    starts = [stamp for stamp in (_event_start(event) for event in selected) if stamp is not None]
    fraction, count, span_days = _fraction_from_starts(starts)
    source = (
        f"{path}: {EVA_FRACTION_FORMULA} "
        f"({count} ISS EVAs / {span_days:.6g} days = {fraction:.6g})"
    )
    return fraction, source


def _results_list(payload: object, path: Path) -> list:
    if isinstance(payload, dict):
        if "results" not in payload:
            raise ValueError(
                f"{path} has no Launch Library results list. "
                "Pass eva_day_fraction (0 to 1 inclusive) to mission_food."
            )
        results = payload["results"]
    elif isinstance(payload, list):
        results = payload
    else:
        raise ValueError(
            f"{path} is not a Launch Library 2 spacewalks cache. "
            "Pass eva_day_fraction (0 to 1 inclusive) to mission_food."
        )
    if not isinstance(results, list):
        raise ValueError(
            f"{path} results must be a list. Pass eva_day_fraction (0 to 1 inclusive) to mission_food."
        )
    return results


def _iss_events(results: Sequence[object]) -> list[Mapping]:
    """Keep ISS EVAs when ``spacestation.name`` exists; otherwise keep every event."""

    events = [event for event in results if isinstance(event, Mapping)]
    names = [_station_name(event) for event in events]
    if any(name is not None for name in names):
        return [event for event, name in zip(events, names) if name == ISS_STATION_NAME]
    return events


def _station_name(event: Mapping) -> str | None:
    station = event.get("spacestation")
    if not isinstance(station, Mapping) or "name" not in station:
        return None
    name = station.get("name")
    if name is None:
        return None
    text = str(name).strip()
    return text or None


def _event_start(event: Mapping) -> datetime | None:
    for key in _DATE_KEYS:
        parsed = _parse_time(event.get(key))
        if parsed is not None:
            return parsed
    return None


def _parse_time(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _fraction_from_starts(starts: Sequence[datetime]) -> tuple[float, int, float]:
    count = len(starts)
    if count == 0:
        return 0.0, 0, 0.0
    span_days = (max(starts) - min(starts)).total_seconds() / _DAY_SECONDS
    if span_days <= 0.0:
        return 1.0, count, 0.0
    fraction = count / span_days
    if fraction < 0.0:
        fraction = 0.0
    elif fraction > 1.0:
        fraction = 1.0
    return fraction, count, span_days


def _positive_days(days: int | float) -> float:
    try:
        value = float(days)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"days must be positive, got {days!r}") from exc
    if isinstance(days, bool) or not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"days must be positive, got {days!r}")
    return value


def _non_negative_margin(safety_margin: float) -> float:
    try:
        value = float(safety_margin)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"safety_margin must be >= 0, got {safety_margin!r}") from exc
    if isinstance(safety_margin, bool) or not math.isfinite(value) or value < 0.0:
        raise ValueError(f"safety_margin must be >= 0, got {safety_margin!r}")
    return value


def _unit_interval(eva_day_fraction: float) -> float:
    try:
        value = float(eva_day_fraction)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"eva_day_fraction must be between 0 and 1 inclusive, got {eva_day_fraction!r}"
        ) from exc
    if isinstance(eva_day_fraction, bool) or not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError(f"eva_day_fraction must be between 0 and 1 inclusive, got {eva_day_fraction!r}")
    return value


def _crew_for_day(crew: Sequence[Mapping] | Mapping, *, eva: bool, eva_hours: float):
    if isinstance(crew, Mapping):
        return _member_for_day(crew, eva=eva, eva_hours=eva_hours)
    if isinstance(crew, (str, bytes)) or not isinstance(crew, Sequence):
        return crew
    updated = []
    for member in crew:
        if isinstance(member, Mapping):
            updated.append(_member_for_day(member, eva=eva, eva_hours=eva_hours))
        else:
            updated.append(member)
    return updated


def _member_for_day(member: Mapping, *, eva: bool, eva_hours: float) -> dict:
    updated = dict(member)
    if not eva:
        updated["eva_hours"] = 0
    elif "eva_hours" not in member:
        updated["eva_hours"] = eva_hours
    return updated


def _infeasible_message(typical: Mapping, eva: Mapping) -> str:
    parts: list[str] = []
    for label, day in (("typical", typical), ("EVA", eva)):
        if day.get("feasible"):
            continue
        unmet = [str(name) for name in day.get("unmet") or []]
        if unmet:
            parts.append(f"Infeasible {label} day. Unmet nutrients: {', '.join(unmet)}.")
        else:
            detail = str(day.get("message") or "Nutrient targets cannot be met.").rstrip(".")
            parts.append(f"Infeasible {label} day. {detail}.")
    if not parts:
        return "Infeasible typical or EVA day."
    return " ".join(parts)


def _packing_list(
    typical: Mapping,
    eva: Mapping,
    typical_days: float,
    eva_days: float,
    safety_margin: float,
) -> list[dict]:
    typical_menu = _menu_index(typical)
    eva_menu = _menu_index(eva)
    scale = 1.0 + safety_margin
    rows: list[dict] = []
    for item in set(typical_menu) | set(eva_menu):
        typical_servings, typical_grams = typical_menu.get(item, (0.0, None))
        eva_servings, eva_grams = eva_menu.get(item, (0.0, None))
        grams = typical_grams if typical_grams is not None else eva_grams
        if grams is None:
            continue
        servings = (typical_servings * typical_days + eva_servings * eva_days) * scale
        if not math.isfinite(servings) or servings <= 0.0:
            continue
        rows.append(
            {
                "item": item,
                "servings": float(servings),
                "mass_kg": float(servings) * float(grams) / 1000.0,
            }
        )
    rows.sort(key=lambda row: (-row["mass_kg"], row["item"]))
    return rows


def _menu_index(day: Mapping) -> dict[str, tuple[float, float]]:
    """Map an item to total servings and packaged grams per serving.

    The optimizer stores mass contribution as servings times the foods-table
    ``mass_g``. Dividing that contribution by the day's servings recovers
    grams per serving.
    """

    totals: dict[str, list[float]] = {}
    for food in day.get("foods") or []:
        item = str(food["item"])
        servings = float(food["servings"])
        if not math.isfinite(servings) or servings <= 0.0:
            continue
        mass_g = float(food["mass_g"])
        slot = totals.setdefault(item, [0.0, 0.0])
        slot[0] += servings
        slot[1] += mass_g
    indexed: dict[str, tuple[float, float]] = {}
    for item, (servings, mass_g) in totals.items():
        indexed[item] = (servings, mass_g / servings)
    return indexed
