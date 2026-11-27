from html.parser import HTMLParser

import pytest

from src import allocation, overflow
from src.dispatch import brief_html, build_notices, sms, whatsapp
from src.predict import build_features

AS_OF = "2025-09-15"
ARGS = (AS_OF, 2, 60, 20, 70)


@pytest.fixture(scope="module")
def features():
    return build_features()


@pytest.fixture(scope="module")
def english(features):
    return build_notices(features, *ARGS)


def test_sms_lengths_and_critical_alerts(english):
    assert any(notice["critical"] for notice in english)
    for notice in english:
        message = sms(notice)
        assert len(message) <= 160
        if notice["critical"]:
            assert "CRITICAL DEFICIT" in message
            assert "*CRITICAL DEFICIT*" in whatsapp(notice)


def test_numbers_match_planners(features, english):
    allocated = allocation.plan(features, AS_OF, 2, 60, 20).to_dict("records")
    routed = overflow.overflow_plan(features, AS_OF, 2, beds_available_pct=70)
    assert len(english) == len(allocated) == len(routed["wards"]) == 60
    assert routed["transfers"]
    for notice, row in zip(english, allocated):
        ward_id = row["ward_id"]
        for key, value in row.items():
            if key == "critical":
                assert notice["allocation_critical"] == value
            elif key == "forecast_week":
                assert notice[key] == str(value)
            else:
                assert notice[key] == value
        ward_routing = routed["wards"][ward_id]
        assert notice["overflow"] == ward_routing
        assert notice["transfers"] == [
            transfer for transfer in routed["transfers"]
            if ward_id in (transfer["from_ward"], transfer["to_ward"])
        ]
        assert notice["unroutable"] == ward_routing["net_deficit"]
        assert notice["critical"] == bool(row["critical"] or ward_routing["net_deficit"])


def test_localized_output_is_offline_and_deterministic(features, english):
    bangla = build_notices(features, *ARGS, "bn")
    assert "রোগী" in brief_html(bangla)
    assert "ঝুঁকি" in sms(bangla[0])
    assert "ঘাটতি" in whatsapp(bangla[0])
    for lang, notices in (("en", english), ("bn", bangla)):
        def formats(rows):
            return [*map(sms, rows), *map(whatsapp, rows), brief_html(rows)]

        output = formats(notices)
        assert output == formats(build_notices(features, *ARGS, lang))
        assert all("http" not in body.lower() for body in output)


def test_printable_brief_contains_one_section_per_ward_critical_first(english):
    class Sections(HTMLParser):
        def __init__(self):
            super().__init__()
            self.classes = []

        def handle_starttag(self, tag, attrs):
            if tag == "section":
                self.classes.append(dict(attrs)["class"])

    parser = Sections()
    page = brief_html(list(reversed(english)))
    parser.feed(page)
    critical = sum(notice["critical"] for notice in english)
    assert 0 < critical < len(english)
    assert parser.classes == ["critical"] * critical + ["routine"] * (len(english) - critical)
    assert "@media print" in page


def test_optional_intervention_keeps_baseline_and_renders_scenario(features, english):
    from src.intervention import simulate

    ward_id = english[0]["ward_id"]
    result = simulate(features, AS_OF, {ward_id: 1})
    notice = build_notices(features, *ARGS, intervention_results=result)[0]
    scenario = result["wards"][0]["by_horizon"]["2"]
    assert notice["intervention"] == {
        "crews": 1, "cases_before": scenario["cases_before"], "cases_after": scenario["cases_after"],
    }
    assert notice["predicted_cases"] == english[0]["predicted_cases"]
    assert "Spraying simulation" in whatsapp(notice)
    assert "Spraying simulation" in brief_html([notice])


@pytest.mark.parametrize(("index", "value", "message"), [
    (0, "invalid", "valid Monday"),
    (1, 5, "Horizon"),
    (2, 29, "Stock"),
    (3, float("inf"), "whole number"),
    (4, 101, "Beds available"),
])
def test_bad_inputs_raise_value_error(features, index, value, message):
    args = list(ARGS)
    args[index] = value
    with pytest.raises(ValueError, match=message):
        build_notices(features, *args)


def test_invalid_language_raises_value_error(features):
    with pytest.raises(ValueError, match="English"):
        build_notices(features, *ARGS, "es")
