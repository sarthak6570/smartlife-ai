from ..config.financial_constants import DEFAULT_DEBT_RETURN, DEFAULT_EQUITY_RETURN, SAFE_WITHDRAWAL_RATE


def project_retirement(
    current_age: int,
    retirement_age: int,
    life_expectancy: int,
    current_corpus: float,
    monthly_sip: float,
    sip_step_up: float,
    annual_expenses: float,
    inflation: float,
    equity_return: float = DEFAULT_EQUITY_RETURN,
    debt_return: float = DEFAULT_DEBT_RETURN,
) -> dict:
    if current_age < 0:
        raise ValueError("Current age cannot be negative.")
    if retirement_age <= current_age:
        raise ValueError("Retirement age must be greater than current age.")
    if life_expectancy < retirement_age:
        raise ValueError("Life expectancy must be greater than or equal to retirement age.")
    for name, value in {
        "current_corpus": current_corpus,
        "monthly_sip": monthly_sip,
        "sip_step_up": sip_step_up,
        "annual_expenses": annual_expenses,
        "inflation": inflation,
        "equity_return": equity_return,
        "debt_return": debt_return,
    }.items():
        if value < 0:
            raise ValueError(f"{name} cannot be negative.")

    corpus, sip, rows = float(current_corpus), float(monthly_sip), []
    for age in range(current_age, retirement_age):
        equity_weight = max(0.35, min(0.80, 0.80 - max(0, age - 40) * 0.015))
        annual_return = equity_weight * equity_return + (1 - equity_weight) * debt_return
        corpus = corpus * (1 + annual_return) + sip * 12
        rows.append({"age": age + 1, "phase": "accumulation", "corpus": round(corpus, 2), "annual_expenses": round(annual_expenses * (1 + inflation) ** (age + 1 - current_age), 2), "equity_weight": round(equity_weight * 100, 1)})
        sip *= 1 + sip_step_up
    required_corpus = annual_expenses * (1 + inflation) ** (retirement_age - current_age) / SAFE_WITHDRAWAL_RATE
    projected_corpus = corpus
    ran_out = False
    retirement_expense = annual_expenses * (1 + inflation) ** (retirement_age - current_age)
    for age in range(retirement_age, life_expectancy):
        corpus = corpus * (1 + debt_return) - retirement_expense
        if corpus < 0:
            ran_out = True
            corpus = 0
        rows.append({"age": age + 1, "phase": "drawdown", "corpus": round(corpus, 2), "annual_expenses": round(retirement_expense, 2), "equity_weight": 0})
        retirement_expense *= 1 + inflation
    return {"required_corpus": round(required_corpus, 2), "projected_corpus_at_retirement": round(projected_corpus, 2), "readiness_percent": round(projected_corpus / required_corpus * 100, 2) if required_corpus else 0, "shortfall_or_surplus": round(projected_corpus - required_corpus, 2), "ran_out_of_money": ran_out, "yearly_projection": rows}
