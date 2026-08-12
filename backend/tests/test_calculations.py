import pytest

from app.calculations.budget import calculate_budget
from app.calculations.retirement import project_retirement


def test_budget_savings_rate():
    assert calculate_budget(100000, 70000)["savings_rate"] == 30


def test_retirement_projection_has_timeline():
    result = project_retirement(30, 60, 85, 0, 25000, .1, 600000, .06)
    assert result["yearly_projection"]
    assert "ran_out_of_money" in result


def test_retirement_projection_rejects_invalid_age_ranges():
    with pytest.raises(ValueError, match="Retirement age must be greater than current age"):
        project_retirement(60, 60, 85, 0, 25000, .1, 600000, .06)

    with pytest.raises(ValueError, match="Life expectancy must be greater than or equal to retirement age"):
        project_retirement(30, 60, 59, 0, 25000, .1, 600000, .06)
