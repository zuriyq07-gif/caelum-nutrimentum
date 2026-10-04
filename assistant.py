"""Mission assistant tools and the xAI chat-completions tool loop.

Chat completions, confirmed 2026-10-04:

* POST https://api.x.ai/v1/chat/completions
* Model ``grok-4.7``
* Docs: https://docs.x.ai/developers/model-capabilities/legacy/chat-completions
* Tool calling: https://docs.x.ai/developers/tools/function-calling

The OpenAI-compatible tool schema is nested under ``function``. The model
returns ``tool_calls``; this module runs them locally and sends the tool
result back on the next request. ``MissionState`` is the only copy of the
mission inputs. Mutating tools call the injected planner, which the Streamlit
page points at the cached ``mission_food`` entry.

``add_eva_hours`` adds to the current weekly total. If that sum would be
negative, the tool rejects the change and leaves the current value in place.
"""

from __future__ import annotations

import json
import math
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

from decay import _below_minimum, describe_shortfalls

CHAT_COMPLETIONS_URL = "https://api.x.ai/v1/chat/completions"
CHAT_MODEL = "grok-4.7"
PACKING_DIFF_LIMIT = 8
MAX_CREW = 6
MAX_MISSION_DAYS = 1000
MAX_EVA_HOURS_PER_WEEK = 80.0
MIN_CREW_AGE = 19
MAX_CREW_AGE = 100

# Same first-load crew the sidebar used before the assistant. Extra seats
# still come from this list when the crew count grows in the form.
DEFAULT_CREW = (
    {"age": 45, "sex": "Male", "weight_kg": 82.9, "height_cm": 180},
    {"age": 38, "sex": "Female", "weight_kg": 62.0, "height_cm": 168},
    {"age": 41, "sex": "Male", "weight_kg": 77.0, "height_cm": 178},
    {"age": 34, "sex": "Female", "weight_kg": 58.0, "height_cm": 165},
    {"age": 50, "sex": "Male", "weight_kg": 85.0, "height_cm": 182},
    {"age": 29, "sex": "Female", "weight_kg": 60.0, "height_cm": 170},
)

SYSTEM_PROMPT = (
    "You are the mission food assistant for this ISS food manifest. "
    "Use tools for any change and for any numeric question about mass, the packing list, "
    "or nutrient shortfalls. Report only figures that appear in tool results. "
    "If a tool returns infeasible, name the nutrients from that result and do not invent a mass. "
    "If required crew fields are missing, ask for them instead of inventing a person. "
    "Mission length is the flight in days. Resupply delay is extra days the pantry must cover "
    "because resupply arrives later; it is not a new mission length. "
    "add_eva_hours adds to the current EVA hours per week. "
    "It rejects the change when the result would be negative, and the current value stays. "
    "Crew indexes are 1-based, matching the sidebar. "
    "Sex is male or female. Height is in centimeters."
)

Planner = Callable[["MissionState"], Mapping]
Replanner = Callable[["MissionState"], Mapping]


class AssistantError(Exception):
    """A failure the page can show without a traceback."""


class ToolRejection(Exception):
    """The tool refused the change. Mission state is left as it was."""


@dataclass
class CrewMember:
    age: int
    sex: str
    weight_kg: float
    height_cm: float
    allergies: list[str] = field(default_factory=list)

    def copy(self) -> CrewMember:
        return CrewMember(
            age=int(self.age),
            sex=str(self.sex),
            weight_kg=float(self.weight_kg),
            height_cm=float(self.height_cm),
            allergies=list(self.allergies),
        )


@dataclass
class MissionState:
    """Single copy of the mission inputs the sidebar and the assistant share."""

    mission_days: int = 30
    crew: list[CrewMember] = field(default_factory=list)
    eva_hours_per_week: float = 6.5
    safety_margin: float = 0.10
    resupply_delay_days: int = 0

    @property
    def packed_days(self) -> int:
        """Days of food to pack: the flight plus the resupply delay."""

        return int(self.mission_days) + int(self.resupply_delay_days)

    @property
    def storage_offset_days(self) -> int:
        """Storage age added to each mission day. Equals the resupply delay."""

        return int(self.resupply_delay_days)

    def copy(self) -> MissionState:
        return MissionState(
            mission_days=int(self.mission_days),
            crew=[member.copy() for member in self.crew],
            eva_hours_per_week=float(self.eva_hours_per_week),
            safety_margin=float(self.safety_margin),
            resupply_delay_days=int(self.resupply_delay_days),
        )

    def crew_cache_key(self) -> tuple[tuple[int, str, float, float, tuple[str, ...]], ...]:
        """Cache key shared with the manifest. Height is meters, as the planner stores it."""

        return tuple(
            (
                int(member.age),
                str(member.sex),
                float(member.weight_kg),
                float(member.height_cm) / 100.0,
                tuple(member.allergies),
            )
            for member in self.crew
        )


def default_mission_state() -> MissionState:
    first = DEFAULT_CREW[0]
    return MissionState(
        mission_days=30,
        crew=[
            CrewMember(
                age=int(first["age"]),
                sex=str(first["sex"]).strip().lower(),
                weight_kg=float(first["weight_kg"]),
                height_cm=float(first["height_cm"]),
                allergies=[],
            )
        ],
        eva_hours_per_week=6.5,
        safety_margin=0.10,
        resupply_delay_days=0,
    )


def resupply_delay_line(days: int) -> str:
    return f"Resupply delayed {int(days)} days"


def resolve_api_key() -> str | None:
    """Environment ``XAI_API_KEY``, otherwise Streamlit secret ``xai_api_key``."""

    env = os.environ.get("XAI_API_KEY", "").strip()
    if env:
        return env
    try:
        import streamlit as st

        if "xai_api_key" not in st.secrets:
            return None
        value = str(st.secrets["xai_api_key"]).strip()
    except Exception:
        return None
    return value or None


def _object_schema(properties: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": properties, "required": required}


def _function_tool(name: str, description: str, parameters: dict) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }


TOOLS: list[dict] = [
    _function_tool(
        "set_mission_length",
        "Set the mission length in days and pack the menu again. "
        "Days is an integer from 1 to 1000. This does not change the resupply delay.",
        _object_schema(
            {"days": {"type": "integer", "description": "Mission length in days."}},
            ["days"],
        ),
    ),
    _function_tool(
        "add_crew",
        "Append one crew member and pack the menu again. "
        "Required: age (years, 19 or older), sex (male or female), weight_kg, and height_cm. "
        "Allergies are optional. Do not invent a person when a required field is missing; ask for it. "
        f"The manifest holds at most {MAX_CREW} crew.",
        _object_schema(
            {
                "age": {"type": "integer", "description": "Age in years."},
                "sex": {"type": "string", "enum": ["male", "female"], "description": "male or female."},
                "weight_kg": {"type": "number", "description": "Body mass in kilograms."},
                "height_cm": {"type": "number", "description": "Height in centimeters, not meters."},
                "allergies": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional allergen words, such as almond or shrimp.",
                },
            },
            ["age", "sex", "weight_kg", "height_cm"],
        ),
    ),
    _function_tool(
        "remove_crew",
        "Remove one crew member by the 1-based index shown in the sidebar, then pack again. "
        "An index that is not on the crew is rejected and nothing changes. "
        "The last crew member cannot be removed.",
        _object_schema(
            {"index": {"type": "integer", "description": "1-based crew index from the sidebar."}},
            ["index"],
        ),
    ),
    _function_tool(
        "add_eva_hours",
        "Add this many hours to the current EVA hours per week, then pack again. "
        "This does not replace the current value. "
        "If the sum would be negative, reject the change and leave the current hours unchanged. "
        f"The result also has to stay at or below {MAX_EVA_HOURS_PER_WEEK:g} hours per week.",
        _object_schema(
            {
                "hours": {
                    "type": "number",
                    "description": "Hours to add to the current EVA hours per week. May be fractional.",
                }
            },
            ["hours"],
        ),
    ),
    _function_tool(
        "delay_resupply",
        "Add N days to the resupply delay. N is an integer of at least 1. "
        "This is not set_mission_length. The pantry is packed for mission days plus the delay, "
        "because resupply arrives later. Food eaten on mission day t uses storage age t plus the delay.",
        _object_schema(
            {"days": {"type": "integer", "description": "Days to add to the current resupply delay."}},
            ["days"],
        ),
    ),
    _function_tool(
        "nutrient_shortfalls",
        "Report vitamin shortfalls on the locked day-0 menu for the current mission. "
        "Does not change the mission. Include the vitamin, the first mission day below target, "
        "how many days are short, and whether the replanned menu meets the target at each epoch start.",
        _object_schema({}, []),
    ),
]

_TOOL_NAMES = {tool["function"]["name"] for tool in TOOLS}


def diff_packing(
    previous_list: Sequence[Mapping] | None,
    new_list: Sequence[Mapping] | None,
    previous_total_mass_kg: object,
    new_total_mass_kg: object,
) -> dict:
    """Added, removed, and changed packing rows, largest absolute mass change first.

    Unchanged items are omitted. At most ``PACKING_DIFF_LIMIT`` rows are returned.
    ``packing_changes_omitted`` is how many changed rows were left off that list.
    Mass totals come from the planner figures passed in, not from a second sum.
    """

    previous = _rows_by_item(previous_list)
    current = _rows_by_item(new_list)
    changes: list[dict] = []
    for item in set(previous) | set(current):
        old = previous.get(item)
        new = current.get(item)
        old_servings = 0.0 if old is None else float(old["servings"])
        old_mass = 0.0 if old is None else float(old["mass_kg"])
        new_servings = 0.0 if new is None else float(new["servings"])
        new_mass = 0.0 if new is None else float(new["mass_kg"])
        if old is None:
            kind = "added"
        elif new is None:
            kind = "removed"
        elif abs(new_servings - old_servings) <= 1e-4 and abs(new_mass - old_mass) <= 1e-6:
            continue
        else:
            kind = "changed"
        changes.append(
            {
                "item": item,
                "change": kind,
                "previous_servings": old_servings,
                "servings": new_servings,
                "previous_mass_kg": old_mass,
                "mass_kg": new_mass,
                "mass_delta_kg": new_mass - old_mass,
            }
        )
    changes.sort(key=lambda row: (-abs(float(row["mass_delta_kg"])), str(row["item"])))
    shown = changes[:PACKING_DIFF_LIMIT]
    omitted = len(changes) - len(shown)
    previous_total = _finite_or_none(previous_total_mass_kg)
    new_total = _finite_or_none(new_total_mass_kg)
    delta = None if previous_total is None or new_total is None else new_total - previous_total
    note = None
    if omitted:
        note = f"{len(shown)} changes listed, {omitted} more not shown."
    return {
        "previous_total_mass_kg": previous_total,
        "total_mass_kg": new_total,
        "mass_delta_kg": delta,
        "packing_changes": shown,
        "packing_changes_omitted": omitted,
        "packing_note": note,
    }


def execute_tool(
    name: str,
    arguments: Mapping | str | None,
    state: MissionState,
    *,
    plan: Planner,
    replan: Replanner,
) -> tuple[MissionState, dict]:
    """Run one tool. Rejections return the same state object and an error payload.

    Mutating tools call ``plan`` on the previous state and the updated state.
    ``nutrient_shortfalls`` calls ``replan`` and does not change ``state``.
    """

    try:
        args = _arguments(arguments)
        if name not in _TOOL_NAMES:
            raise ToolRejection(f"Unknown tool {name}.")
        if name == "nutrient_shortfalls":
            return state, _shortfalls(state, replan)
        updated = _mutate(name, args, state)
    except ToolRejection as exc:
        return state, {"ok": False, "error": str(exc), "mutated": False}
    try:
        before = plan(state)
        after = plan(updated)
    except Exception as exc:
        return state, {
            "ok": False,
            "error": f"The planner stopped. {type(exc).__name__}: {exc}",
            "mutated": False,
        }
    return updated, _plan_payload(updated, before, after)


def run_turn(
    messages: Sequence[Mapping],
    state: MissionState,
    api_key: str,
    *,
    plan: Planner,
    replan: Replanner,
    max_rounds: int = 6,
) -> tuple[list[dict], MissionState, list[dict]]:
    """Call xAI until the assistant message has no further tool calls.

    Returns the conversation (without the system prompt), the state after
    every tool that ran, and the raw tool payloads in call order.
    """

    key = str(api_key or "").strip()
    if not key:
        raise AssistantError(
            "The assistant needs XAI_API_KEY. Set that environment variable, "
            "or set xai_api_key in Streamlit secrets."
        )
    conversation = []
    for message in messages:
        wire = _public_message(message)
        if message.get("error"):
            wire["error"] = True
        conversation.append(wire)
    summaries: list[dict] = []
    for _round in range(max_rounds):
        payload = {
            "model": CHAT_MODEL,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}, *_api_messages(conversation)],
            "tools": TOOLS,
            "tool_choice": "auto",
        }
        data = post_json(CHAT_COMPLETIONS_URL, payload, key)
        message = _choice_message(data)
        tool_calls = _tool_calls(message)
        conversation.append(_assistant_wire(message, tool_calls))
        if not tool_calls:
            if not _message_text(message.get("content")).strip():
                raise AssistantError("The assistant returned an empty reply.")
            return conversation, state, summaries
        for call in tool_calls:
            name = str(call["function"]["name"])
            state, summary = execute_tool(
                name,
                call["function"].get("arguments"),
                state,
                plan=plan,
                replan=replan,
            )
            summaries.append({"tool": name, **summary})
            conversation.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": json.dumps(summary),
                }
            )
    raise AssistantError("The assistant used tools several times and did not finish an answer.")


def post_json(url: str, payload: Mapping, api_key: str, *, timeout: float = 120.0) -> dict:
    """POST JSON to the xAI API. The key is sent only on the Authorization header."""

    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    raw = _read_request(request, timeout)
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError:
        raise AssistantError("The xAI API returned a response that was not JSON.") from None
    if not isinstance(parsed, dict):
        raise AssistantError("The xAI API returned a response that was not a JSON object.")
    return parsed


def _read_request(request: urllib.request.Request, timeout: float) -> bytes:
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        detail = _public_error(_error_text(exc.read()))
        if exc.code in {401, 403}:
            raise AssistantError(
                f"The xAI API rejected the key ({exc.code}). Check XAI_API_KEY. {detail}".rstrip()
            ) from None
        raise AssistantError(f"The xAI API returned {exc.code}. {detail}".rstrip()) from None
    except urllib.error.URLError as exc:
        reason = _public_error(str(exc.reason))
        raise AssistantError(f"The xAI API could not be reached. {reason}") from None


def _shortfalls(state: MissionState, replan: Replanner) -> dict:
    try:
        decay = replan(state)
    except Exception as exc:
        return {
            "ok": False,
            "error": f"Shelf-life replanning stopped. {type(exc).__name__}: {exc}",
            "mutated": False,
        }
    if not isinstance(decay, Mapping):
        return {"ok": False, "error": "Shelf-life replanning returned nothing usable.", "mutated": False}
    rows = []
    daily = list(decay.get("daily") or [])
    starts = [int(day) for day in (decay.get("epoch_starts") or [])]
    for shortfall in decay.get("shortfalls") or []:
        if not isinstance(shortfall, Mapping):
            continue
        vitamin = str(shortfall.get("vitamin") or "")
        epochs = []
        for day in starts:
            match = next(
                (
                    row
                    for row in daily
                    if isinstance(row, Mapping) and str(row.get("vitamin") or "") == vitamin and int(row.get("day") or 0) == day
                ),
                None,
            )
            if match is None or match.get("replan_total") is None or match.get("target_min") is None:
                meets = None
            else:
                meets = not _below_minimum(float(match["replan_total"]), float(match["target_min"]))
            epochs.append({"day": day, "replan_meets_target": meets})
        rows.append(
            {
                "vitamin": vitamin,
                "label": shortfall.get("label") or vitamin,
                "unit": shortfall.get("unit") or "",
                "first_day": shortfall.get("first_day"),
                "days_short": shortfall.get("days_short"),
                "replan_at_epoch_starts": epochs,
            }
        )
    message = decay.get("message") or describe_shortfalls(decay.get("shortfalls") or [])
    return {
        "ok": True,
        "mutated": False,
        "mission_days": state.mission_days,
        "resupply_delay_days": state.resupply_delay_days,
        "storage_offset_days": decay.get("storage_offset_days", state.storage_offset_days),
        "epoch_starts": starts,
        "shortfalls": rows,
        "message": message,
    }


def _plan_payload(state: MissionState, before: Mapping, after: Mapping) -> dict:
    feasible = bool(after.get("feasible"))
    facts = _state_facts(state)
    if not feasible:
        unmet = _unmet(after)
        return {
            "ok": False,
            "mutated": True,
            "feasible": False,
            "unmet": unmet,
            "unmet_labels": _nutrient_labels(unmet),
            "previous_total_mass_kg": _finite_or_none(before.get("total_mass_kg")),
            "total_mass_kg": None,
            "mass_delta_kg": None,
            "packing_changes": [],
            "packing_changes_omitted": 0,
            "packing_note": "No mass was packed because the menu is infeasible.",
            "message": str(after.get("message") or "This crew and menu cannot meet the nutrient targets."),
            **facts,
        }
    diff = diff_packing(
        list(before.get("packing_list") or []),
        list(after.get("packing_list") or []),
        before.get("total_mass_kg"),
        after.get("total_mass_kg"),
    )
    return {
        "ok": True,
        "mutated": True,
        "feasible": True,
        "message": str(after.get("message") or ""),
        **diff,
        **facts,
    }


def _mutate(name: str, args: Mapping, state: MissionState) -> MissionState:
    if name == "set_mission_length":
        return _set_length(state, args.get("days"))
    if name == "add_crew":
        return _add_crew(state, args)
    if name == "remove_crew":
        return _remove_crew(state, args.get("index"))
    if name == "add_eva_hours":
        return _add_eva(state, args.get("hours"))
    if name == "delay_resupply":
        return _delay(state, args.get("days"))
    raise ToolRejection(f"Unknown tool {name}.")


def _set_length(state: MissionState, days: object) -> MissionState:
    count = _require_int(days, "Mission length", minimum=1)
    if count > MAX_MISSION_DAYS:
        raise ToolRejection(f"Mission length cannot be above {MAX_MISSION_DAYS} days.")
    updated = state.copy()
    updated.mission_days = count
    return updated


def _add_crew(state: MissionState, args: Mapping) -> MissionState:
    missing = [
        field_name
        for field_name in ("age", "sex", "weight_kg", "height_cm")
        if field_name not in args or args.get(field_name) in (None, "")
    ]
    if missing:
        joined = ", ".join(missing)
        raise ToolRejection(f"Need {joined} before adding a crew member. Ask for them.")
    if len(state.crew) >= MAX_CREW:
        raise ToolRejection(f"This manifest holds {MAX_CREW} crew. Remove one before adding another.")
    age = _require_int(args.get("age"), "Age", minimum=MIN_CREW_AGE)
    if age > MAX_CREW_AGE:
        raise ToolRejection(f"Age cannot be above {MAX_CREW_AGE}.")
    sex = _normalize_sex(args.get("sex"))
    weight = _require_float(args.get("weight_kg"), "Weight")
    height = _require_float(args.get("height_cm"), "Height")
    if weight <= 0 or weight > 250:
        raise ToolRejection("Weight has to be above 0 kg and at most 250 kg.")
    if height <= 0 or height > 250:
        raise ToolRejection("Height is in centimeters and has to be above 0 and at most 250.")
    updated = state.copy()
    updated.crew.append(
        CrewMember(
            age=age,
            sex=sex,
            weight_kg=weight,
            height_cm=height,
            allergies=_allergies(args.get("allergies")),
        )
    )
    return updated


def _remove_crew(state: MissionState, index: object) -> MissionState:
    if len(state.crew) <= 1:
        raise ToolRejection("Keep at least one crew member.")
    slot = _require_int(index, "Crew index", minimum=1)
    if slot > len(state.crew):
        raise ToolRejection(
            f"Crew member {slot} is not on this mission. Crew are numbered 1 to {len(state.crew)}."
        )
    updated = state.copy()
    del updated.crew[slot - 1]
    return updated


def _add_eva(state: MissionState, hours: object) -> MissionState:
    added = _require_float(hours, "EVA hours")
    result = float(state.eva_hours_per_week) + added
    if result < 0:
        raise ToolRejection(
            "EVA hours per week cannot go below 0. "
            f"Adding {added:g} to {state.eva_hours_per_week:g} was rejected, and the current value was left unchanged."
        )
    if result > MAX_EVA_HOURS_PER_WEEK:
        raise ToolRejection(
            f"EVA hours per week cannot go above {MAX_EVA_HOURS_PER_WEEK:g}. "
            "The current value was left unchanged."
        )
    updated = state.copy()
    updated.eva_hours_per_week = result
    return updated


def _delay(state: MissionState, days: object) -> MissionState:
    extra = _require_int(days, "Resupply delay", minimum=1)
    updated = state.copy()
    updated.resupply_delay_days = int(state.resupply_delay_days) + extra
    return updated


def _state_facts(state: MissionState) -> dict:
    return {
        "mission_days": int(state.mission_days),
        "resupply_delay_days": int(state.resupply_delay_days),
        "packed_days": int(state.packed_days),
        "storage_offset_days": int(state.storage_offset_days),
        "eva_hours_per_week": float(state.eva_hours_per_week),
        "safety_margin": float(state.safety_margin),
        "crew_count": len(state.crew),
        "crew": [
            {
                "index": index,
                "age": int(member.age),
                "sex": member.sex,
                "weight_kg": float(member.weight_kg),
                "height_cm": float(member.height_cm),
                "allergies": list(member.allergies),
            }
            for index, member in enumerate(state.crew, start=1)
        ],
    }


def _unmet(result: Mapping) -> list[str]:
    names: list[str] = []
    for key in ("typical_day", "eva_day"):
        day = result.get(key) or {}
        if not isinstance(day, Mapping):
            continue
        for name in day.get("unmet") or []:
            text = str(name)
            if text not in names:
                names.append(text)
    return names


def _nutrient_labels(names: Sequence[str]) -> list[str]:
    try:
        from app import nutrient_label
    except Exception:
        return [name.replace("_", " ") for name in names]
    return [nutrient_label(name) for name in names]


def _rows_by_item(rows: Sequence[Mapping] | None) -> dict[str, Mapping]:
    mapped: dict[str, Mapping] = {}
    for row in rows or []:
        if not isinstance(row, Mapping) or "item" not in row:
            continue
        name = str(row.get("item") or "").strip()
        if name:
            mapped[name] = row
    return mapped


def _arguments(raw: Mapping | str | None) -> dict:
    if raw is None or raw == "":
        return {}
    if isinstance(raw, Mapping):
        return dict(raw)
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ToolRejection("The tool arguments were not valid JSON.") from exc
        if not isinstance(parsed, dict):
            raise ToolRejection("The tool arguments have to be an object.")
        return parsed
    raise ToolRejection("The tool arguments have to be an object.")


def _choice_message(data: Mapping) -> Mapping:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise AssistantError("The xAI API returned a chat response without a message.")
    message = choices[0].get("message") if isinstance(choices[0], Mapping) else None
    if not isinstance(message, Mapping):
        raise AssistantError("The xAI API returned a chat response without a message.")
    return message


def _tool_calls(message: Mapping) -> list[dict]:
    raw = message.get("tool_calls") or []
    if not isinstance(raw, list):
        return []
    calls = []
    for call in raw:
        if not isinstance(call, Mapping):
            continue
        function = call.get("function") or {}
        if not isinstance(function, Mapping):
            continue
        arguments = function.get("arguments")
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments if arguments is not None else {})
        calls.append(
            {
                "id": str(call.get("id") or ""),
                "type": "function",
                "function": {"name": str(function.get("name") or ""), "arguments": arguments},
            }
        )
    return calls


def _assistant_wire(message: Mapping, tool_calls: list[dict]) -> dict:
    content = message.get("content")
    wire: dict = {"role": "assistant", "content": content if isinstance(content, str) or content is None else _message_text(content)}
    if tool_calls:
        wire["tool_calls"] = tool_calls
    return wire


def _api_messages(conversation: Sequence[Mapping]) -> list[dict]:
    """Drop page-only keys so the API payload stays the chat-completions shape."""

    cleaned = []
    for message in conversation:
        cleaned.append({key: value for key, value in message.items() if key != "error"})
    return cleaned


def _public_message(message: Mapping) -> dict:
    role = str(message.get("role") or "user")
    if role == "tool":
        return {
            "role": "tool",
            "tool_call_id": str(message.get("tool_call_id") or ""),
            "content": str(message.get("content") or ""),
        }
    if role == "assistant" and message.get("tool_calls"):
        return {
            "role": "assistant",
            "content": message.get("content"),
            "tool_calls": list(message.get("tool_calls") or []),
        }
    return {"role": role, "content": _message_text(message.get("content"))}


def _message_text(content: object) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, Mapping):
                parts.append(str(item.get("text") or ""))
        return "".join(parts)
    return str(content)


def _normalize_sex(value: object) -> str:
    text = str(value or "").strip().lower()
    if text in {"m", "male"}:
        return "male"
    if text in {"f", "female"}:
        return "female"
    raise ToolRejection("Sex has to be male or female.")


def _allergies(value: object) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        parts = value.split(",")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        parts = [str(item) for item in value]
    else:
        raise ToolRejection("Allergies have to be a list of words, or one comma-separated line.")
    cleaned = []
    for part in parts:
        token = str(part).strip()
        if token and token not in cleaned:
            cleaned.append(token)
    return cleaned


def _require_int(value: object, label: str, *, minimum: int) -> int:
    number = _require_float(value, label)
    if not float(number).is_integer():
        raise ToolRejection(f"{label} has to be a whole number.")
    count = int(number)
    if count < minimum:
        if minimum == 1 and "delay" in label.lower():
            raise ToolRejection("Resupply delay must add at least 1 day.")
        if minimum == 1 and "mission" in label.lower():
            raise ToolRejection("Mission length has to be at least 1 day.")
        raise ToolRejection(f"{label} has to be at least {minimum}.")
    return count


def _require_float(value: object, label: str) -> float:
    if isinstance(value, bool) or value is None or value == "":
        raise ToolRejection(f"{label} has to be a number.")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ToolRejection(f"{label} has to be a number.") from None
    if not math.isfinite(number):
        raise ToolRejection(f"{label} has to be a finite number.")
    return number


def _finite_or_none(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _error_text(body: bytes) -> str:
    text = body.decode("utf-8", errors="replace").strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return text[:500]
    if isinstance(parsed, dict):
        err = parsed.get("error")
        if isinstance(err, dict) and err.get("message"):
            return str(err["message"])[:500]
        if isinstance(err, str):
            return err[:500]
    return text[:500]


def _public_error(text: str) -> str:
    import re

    return re.sub(r"Bearer\s+\S+", "Bearer [redacted]", text)
