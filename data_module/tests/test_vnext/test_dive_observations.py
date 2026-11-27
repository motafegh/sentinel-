import csv
import io
import pytest

from sentinel_data.vnext.dive_observations import (
    COLUMNS, CATEGORIES, observation_rows, published_labels, category_view,
    tool_state, assert_optimizer_reservations, pick_groups,
)


def make_csv(fields, records):
    out = io.StringIO(newline="")
    writer = csv.DictWriter(out, fieldnames=["contractID", *fields])
    writer.writeheader()
    writer.writerows(records)
    return out.getvalue().encode()


def test_literal_rows_and_complementary_observations():
    data = make_csv(COLUMNS, [
        {"contractID": "1", "MAIAN_1": "0"},
        {"contractID": "1", "Slither_1": "1"},
    ])
    rows = list(observation_rows(data))
    assert rows == list(observation_rows(data))
    assert rows[0]["row_id"] != rows[1]["row_id"]
    assert rows[0]["MAIAN_1"] == 0 and rows[0]["Slither_1"] is None
    labels = {"1": dict.fromkeys(CATEGORIES, 0)}
    view = list(category_view(rows, labels))
    assert len(view) == 8
    assert view[0]["MAIAN_state"] == "OBSERVED_ZERO"
    assert view[0]["Slither_state"] == "OBSERVED_ONE"
    assert view[0]["first_row_positive_tools"] == 0
    assert view[0]["positive_observation_tools"] == 1
    assert view[0]["published_annotation"] == 0
    assert view[0]["missing_tools"] == 4
    assert view[1]["missing_tools"] == 6
    assert "target" not in view[0]


@pytest.mark.parametrize("values,state", [
    ([None, None], "MISSING"), ([None, 0, 0], "OBSERVED_ZERO"),
    ([1, None], "OBSERVED_ONE"), ([0, 1, None], "CONFLICTING"),
])
def test_tool_states(values, state):
    assert tool_state(iter(values)) == state


def test_conflict_retained_not_majority_voted():
    rows = list(observation_rows(make_csv(COLUMNS, [
        {"contractID": "1", "Slither_1": "1"},
        {"contractID": "1", "Slither_1": "0"},
        {"contractID": "1", "Slither_1": "0"},
    ])))
    cell = next(category_view(rows, {"1": dict.fromkeys(CATEGORIES, 0)}))
    assert cell["Slither_state"] == "CONFLICTING"
    assert cell["conflicting_tools"] == 1
    assert cell["positive_observation_tools"] == 1
    assert len(cell["observation_row_ids"]) == 3


@pytest.mark.parametrize("cid,value", [("01","0"),("0","1"),("x","1"),("1","2"),("1"," 1")])
def test_invalid_domain(cid, value):
    with pytest.raises(ValueError):
        list(observation_rows(make_csv(COLUMNS,[{"contractID": cid,"Slither_1":value}])))


def test_bad_header_and_missing_field():
    for data in [b"contractID,other\n1,0\n", (",".join(["contractID", *COLUMNS])+"\n1,0\n").encode()]:
        with pytest.raises(ValueError):
            list(observation_rows(data))


def test_labels_unique_binary_and_matching_ids():
    record = {"contractID":"1", **dict.fromkeys(CATEGORIES,"0")}
    with pytest.raises(ValueError):
        published_labels(make_csv(CATEGORIES,[record, record]))
    with pytest.raises(ValueError):
        published_labels(make_csv(CATEGORIES,[{**record, CATEGORIES[0]:""}]))
    with pytest.raises(ValueError):
        list(category_view([], published_labels(make_csv(CATEGORIES,[record]))))


def test_selection_deterministic_unique_and_reserved():
    rows = [{"group_id":g,"contract_id":f"c{i}","native_id":str(i)} for i,g in enumerate(["a","a","b","c","d"])]
    quotas = [("one",2),("two",2)]
    selected = pick_groups({"one":rows,"two":rows}, quotas, {"a"}, "test")
    assert selected == pick_groups({"one":rows[::-1],"two":rows[::-1]}, quotas, {"a"}, "test")
    assert {x["group_id"] for x in selected} == {"b","c","d"}
    assert len(selected) == 3
    assert_optimizer_reservations(["a"], {"b","c","d"})
    with pytest.raises(ValueError):
        assert_optimizer_reservations(["b"], {"b","c","d"})
