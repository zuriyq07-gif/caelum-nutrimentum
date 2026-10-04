"""Mission assistant tools, packing diffs, and the chat tool loop. No network."""

import json
import urllib.request

import pytest

from assistant import (
    CHAT_COMPLETIONS_URL,
    AssistantError,
    MissionState,
    default_mission_state,
    diff_packing,
    execute_tool,
    resupply_delay_line,
    run_turn,
)
from voice import STT_MODEL, STT_URL, TTS_URL, synthesize_speech, transcribe_audio


def _plan(state: MissionState) -> dict:
    mass = float(state.packed_days) + state.eva_hours_per_week
    return {
        "feasible": True,
        "total_mass_kg": mass,
        "packing_list": [{"item": "Rice", "servings": mass, "mass_kg": mass}],
        "message": "ok",
        "days": state.packed_days,
        "typical_day": {"feasible": True, "unmet": []},
        "eva_day": {"feasible": True, "unmet": []},
    }


def _boom(_state):
    raise AssertionError("planner should not run")


def test_set_mission_length_and_reject_non_positive_days():
    state = default_mission_state()
    updated, payload = execute_tool("set_mission_length", {"days": 45}, state, plan=_plan, replan=_boom)
    assert state.mission_days == 30
    assert updated.mission_days == 45
    assert payload["ok"] is True
    assert payload["total_mass_kg"] == pytest.approx(45 + 6.5)
    assert payload["previous_total_mass_kg"] == pytest.approx(30 + 6.5)
    assert payload["mass_delta_kg"] == pytest.approx(15)

    rejected, error = execute_tool("set_mission_length", {"days": 0}, updated, plan=_boom, replan=_boom)
    assert rejected is updated
    assert rejected.mission_days == 45
    assert error["ok"] is False
    negative, error = execute_tool("set_mission_length", {"days": -3}, updated, plan=_boom, replan=_boom)
    assert negative is updated
    assert error["ok"] is False


def test_add_and_remove_crew_including_rejects():
    state = default_mission_state()
    added, payload = execute_tool(
        "add_crew",
        {"age": 38, "sex": "Female", "weight_kg": 62.0, "height_cm": 168, "allergies": ["almond", "shrimp"]},
        state,
        plan=_plan,
        replan=_boom,
    )
    assert len(state.crew) == 1
    assert len(added.crew) == 2
    assert added.crew[1].sex == "female"
    assert added.crew[1].height_cm == 168
    assert added.crew[1].allergies == ["almond", "shrimp"]
    assert payload["crew_count"] == 2

    missing, error = execute_tool(
        "add_crew",
        {"sex": "female", "weight_kg": 60, "height_cm": 165},
        added,
        plan=_boom,
        replan=_boom,
    )
    assert missing is added
    assert "age" in error["error"]
    assert len(added.crew) == 2

    bad_sex, error = execute_tool(
        "add_crew",
        {"age": 30, "sex": "other", "weight_kg": 70, "height_cm": 175},
        added,
        plan=_boom,
        replan=_boom,
    )
    assert bad_sex is added
    assert error["ok"] is False

    out_of_range, error = execute_tool("remove_crew", {"index": 5}, added, plan=_boom, replan=_boom)
    assert out_of_range is added
    assert len(added.crew) == 2
    assert error["ok"] is False
    assert "not on this mission" in error["error"]

    removed, payload = execute_tool("remove_crew", {"index": 1}, added, plan=_plan, replan=_boom)
    assert len(removed.crew) == 1
    assert removed.crew[0].age == 38
    assert payload["mutated"] is True

    last, error = execute_tool("remove_crew", {"index": 1}, removed, plan=_boom, replan=_boom)
    assert last is removed
    assert len(removed.crew) == 1
    assert "one crew member" in error["error"]


def test_add_eva_hours_rejects_a_negative_result():
    state = default_mission_state()
    updated, payload = execute_tool("add_eva_hours", {"hours": 2}, state, plan=_plan, replan=_boom)
    assert state.eva_hours_per_week == 6.5
    assert updated.eva_hours_per_week == pytest.approx(8.5)
    assert payload["eva_hours_per_week"] == pytest.approx(8.5)
    assert payload["mass_delta_kg"] == pytest.approx(2)

    rejected, error = execute_tool("add_eva_hours", {"hours": -10}, updated, plan=_boom, replan=_boom)
    assert rejected is updated
    assert updated.eva_hours_per_week == pytest.approx(8.5)
    assert error["ok"] is False
    assert "below 0" in error["error"]


def test_delay_resupply_adds_to_horizon_and_storage_offset():
    state = default_mission_state()
    delayed, payload = execute_tool("delay_resupply", {"days": 5}, state, plan=_plan, replan=_boom)
    assert state.mission_days == 30
    assert state.resupply_delay_days == 0
    assert delayed.mission_days == 30
    assert delayed.resupply_delay_days == 5
    assert delayed.packed_days == 35
    assert delayed.storage_offset_days == 5
    assert payload["packed_days"] == 35
    assert payload["storage_offset_days"] == 5
    assert payload["previous_total_mass_kg"] == pytest.approx(30 + 6.5)
    assert payload["total_mass_kg"] == pytest.approx(35 + 6.5)
    assert resupply_delay_line(5) == "Resupply delayed 5 days"

    again, payload = execute_tool("delay_resupply", {"days": 2}, delayed, plan=_plan, replan=_boom)
    assert again.resupply_delay_days == 7
    assert again.packed_days == 37
    assert again.storage_offset_days == 7
    assert again.mission_days == 30
    assert payload["packed_days"] == 37

    rejected, error = execute_tool("delay_resupply", {"days": 0}, again, plan=_boom, replan=_boom)
    assert rejected is again
    assert again.resupply_delay_days == 7
    assert error["ok"] is False


def test_packing_diff_lists_changed_rows_by_absolute_mass():
    previous = [
        {"item": "Rice", "servings": 10, "mass_kg": 4.0},
        {"item": "Tea", "servings": 5, "mass_kg": 0.5},
        {"item": "Nuts", "servings": 2, "mass_kg": 1.0},
    ]
    current = [
        {"item": "Rice", "servings": 12, "mass_kg": 4.8},
        {"item": "Beans", "servings": 3, "mass_kg": 2.0},
        {"item": "Nuts", "servings": 2, "mass_kg": 1.0},
    ]
    diff = diff_packing(previous, current, 5.5, 7.8)
    assert diff["previous_total_mass_kg"] == 5.5
    assert diff["total_mass_kg"] == 7.8
    assert diff["mass_delta_kg"] == pytest.approx(2.3)
    assert [row["item"] for row in diff["packing_changes"]] == ["Beans", "Rice", "Tea"]
    assert diff["packing_changes"][0]["change"] == "added"
    assert diff["packing_changes"][1]["change"] == "changed"
    assert diff["packing_changes"][1]["previous_servings"] == 10
    assert diff["packing_changes"][1]["servings"] == 12
    assert diff["packing_changes"][1]["mass_delta_kg"] == pytest.approx(0.8)
    assert diff["packing_changes"][2]["change"] == "removed"
    assert diff["packing_changes_omitted"] == 0
    assert all(row["item"] != "Nuts" for row in diff["packing_changes"])


def test_packing_diff_caps_the_list_and_counts_the_rest():
    previous = []
    current = [{"item": f"Food {index}", "servings": 1, "mass_kg": float(index)} for index in range(1, 11)]
    diff = diff_packing(previous, current, 0, 55)
    assert [row["item"] for row in diff["packing_changes"]] == [f"Food {index}" for index in range(10, 2, -1)]
    assert diff["packing_changes_omitted"] == 2
    assert diff["packing_note"] == "8 changes listed, 2 more not shown."


def test_infeasible_menu_names_nutrients_and_does_not_invent_a_mass():
    def plan(_state):
        return {
            "feasible": False,
            "total_mass_kg": None,
            "packing_list": [],
            "message": "Infeasible.",
            "typical_day": {"feasible": False, "unmet": ["vitamin_d"]},
            "eva_day": {"feasible": True, "unmet": []},
        }

    state = default_mission_state()
    updated, payload = execute_tool("set_mission_length", {"days": 40}, state, plan=plan, replan=_boom)
    assert state.mission_days == 30
    assert updated.mission_days == 40
    assert payload["feasible"] is False
    assert payload["total_mass_kg"] is None
    assert payload["mass_delta_kg"] is None
    assert payload["unmet"] == ["vitamin_d"]
    assert payload["packing_changes"] == []


def test_nutrient_shortfalls_read_the_replan_and_do_not_change_state():
    state = default_mission_state()
    state.resupply_delay_days = 5

    def replan(received):
        assert received.resupply_delay_days == 5
        return {
            "storage_offset_days": 5,
            "epoch_starts": [0],
            "message": "Without replanning, Vitamin A falls short starting on day 0 (30 days).",
            "shortfalls": [
                {
                    "vitamin": "vitamin_a",
                    "label": "Vitamin A",
                    "unit": "ug",
                    "first_day": 0,
                    "days_short": 30,
                }
            ],
            "daily": [
                {
                    "day": 0,
                    "vitamin": "vitamin_a",
                    "replan_total": 100.0,
                    "target_min": 90.0,
                }
            ],
        }

    returned, payload = execute_tool("nutrient_shortfalls", {}, state, plan=_boom, replan=replan)
    assert returned is state
    assert state.mission_days == 30
    assert state.resupply_delay_days == 5
    assert payload["mutated"] is False
    row = payload["shortfalls"][0]
    assert row["vitamin"] == "vitamin_a"
    assert row["first_day"] == 0
    assert row["days_short"] == 30
    assert row["replan_at_epoch_starts"] == [{"day": 0, "replan_meets_target": True}]


def test_run_turn_sends_the_tool_result_on_the_next_request(monkeypatch):
    responses = [
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "set_mission_length", "arguments": "{\"days\": 40}"},
                            }
                        ],
                    }
                }
            ]
        },
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Mission length is 40 days. Total mass is 46.5 kg.",
                    }
                }
            ]
        },
    ]
    seen = []

    def fake_urlopen(request, timeout=None):
        body = json.loads(request.data.decode())
        seen.append((request.full_url, body, request.get_header("Authorization")))
        payload = responses[len(seen) - 1]
        return _Response(json.dumps(payload).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    state = default_mission_state()
    messages, updated, summaries = run_turn(
        [{"role": "user", "content": "Make the mission 40 days."}],
        state,
        "test-key",
        plan=_plan,
        replan=_boom,
    )
    assert seen[0][0] == CHAT_COMPLETIONS_URL
    assert seen[0][2] == "Bearer test-key"
    assert "test-key" not in json.dumps(seen[0][1])
    assert len(seen) == 2
    tool_messages = [message for message in seen[1][1]["messages"] if message["role"] == "tool"]
    assert tool_messages
    tool_body = json.loads(tool_messages[0]["content"])
    assert tool_body["mission_days"] == 40
    assert tool_body["mutated"] is True
    assert updated.mission_days == 40
    assert state.mission_days == 30
    assert summaries[0]["tool"] == "set_mission_length"
    assert messages[-1]["content"] == "Mission length is 40 days. Total mass is 46.5 kg."


def test_run_turn_refuses_to_invent_a_reply_without_a_key():
    with pytest.raises(AssistantError):
        run_turn([], default_mission_state(), "", plan=_boom, replan=_boom)


def test_voice_requests_use_the_documented_endpoints(monkeypatch):
    seen = []

    def fake_urlopen(request, timeout=None):
        seen.append(request)
        if request.full_url == STT_URL:
            return _Response(json.dumps({"text": "Add 2 EVA hours per week", "duration": 1.2}).encode())
        return _Response(b"ID3fake-mp3", headers={"Content-Type": "audio/mpeg"})

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    audio = b"RIFFfake-wav"
    transcript = transcribe_audio(audio, "test-key", filename="question.wav", mime="audio/wav")
    spoken = synthesize_speech("Total mass is 12.0 kg.", "test-key")
    assert transcript == "Add 2 EVA hours per week"
    assert spoken == b"ID3fake-mp3"
    stt = seen[0]
    assert stt.full_url == STT_URL
    assert stt.get_header("Authorization") == "Bearer test-key"
    body = stt.data
    assert STT_MODEL.encode() in body
    assert body.find(STT_MODEL.encode()) < body.find(audio)
    assert b"test-key" not in body
    tts = json.loads(seen[1].data.decode())
    assert seen[1].full_url == TTS_URL
    assert tts["voice_id"] == "eve"
    assert tts["language"] == "en"
    assert tts["text"] == "Total mass is 12.0 kg."
    assert b"test-key" not in seen[1].data


def test_plan_wrappers_use_the_cached_entries_and_eva_schedule(monkeypatch):
    import app

    real_schedule = app.eva_schedule
    packed = {}

    def fake_plan(days, crew, safety_margin, eva_day_fraction, eva_hours):
        packed.update(days=days, fraction=eva_day_fraction, hours=eva_hours, crew=crew, margin=safety_margin)
        return {"feasible": True, "total_mass_kg": 1.0, "packing_list": []}

    monkeypatch.setattr(
        app,
        "eva_schedule",
        lambda hours: {"eva_day_fraction": 0.25, "eva_hours": 3.0, "note": None, "summary": "stub"},
    )
    monkeypatch.setattr(app, "plan_food_load", fake_plan)
    state = default_mission_state()
    state.resupply_delay_days = 5
    state.eva_hours_per_week = 13
    app.plan_for_state(state)
    assert packed["days"] == 35
    assert packed["fraction"] == 0.25
    assert packed["hours"] == 3.0
    assert packed["crew"][0][3] == pytest.approx(1.8)

    shelf = {}

    def fake_shelf(days, crew, safety_margin, eva_day_fraction, eva_hours, replan_every, storage_offset_days=0.0):
        shelf.update(
            days=days,
            fraction=eva_day_fraction,
            hours=eva_hours,
            offset=storage_offset_days,
            every=replan_every,
        )
        return {"shortfalls": [], "daily": [], "epoch_starts": [0]}

    monkeypatch.setattr(app, "plan_shelf_life", fake_shelf)
    monkeypatch.setattr(app, "eva_schedule", real_schedule)
    app.replan_for_state(state)
    assert shelf["days"] == 30
    assert shelf["offset"] == 5
    assert shelf["fraction"] == pytest.approx(2 / 7)
    assert shelf["hours"] == 6.5


class _Response:
    def __init__(self, payload: bytes, headers: dict | None = None):
        self._payload = payload
        self.headers = headers or {"Content-Type": "application/json"}

    def read(self):
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False
