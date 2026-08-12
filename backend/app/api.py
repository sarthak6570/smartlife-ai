from datetime import date
from typing import Any
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from io import BytesIO
from pydantic import BaseModel, Field, field_validator, model_validator
from sqlalchemy.orm import Session
from .auth import current_user
from .database import get_db
from .models import User, Profile, Income, Expense, Asset, Liability, Goal, Investment, RetirementPlan
from .calculations.budget import calculate_budget
from .calculations.wealth import calculate_real_wealth
from .calculations.portfolio import portfolio_summary
from .calculations.life_value import calculate_life_value
from .calculations.retirement import project_retirement
from .ml.expense_prediction import predict_next_expense
from .ml.investment_recommendation import recommend_investments
from .services.tax_service import build_tax_optimization
from .services.profile_service import get_profile as get_user_profile, update_profile as update_user_profile
from .utils.responses import response

router = APIRouter(prefix="/api/v1", tags=["Finance"])
MODELS = {"income": Income, "expenses": Expense, "assets": Asset, "liabilities": Liability, "goals": Goal, "investments": Investment}

class RecordIn(BaseModel):
    name: str
    amount: float = 0
    date: str | None = None
    notes: str | None = None
    frequency: str | None = None
    asset_type: str | None = None
    interest_rate: float | None = None
    target_amount: float | None = None
    target_date: str | None = None
    investment_type: str | None = None
    current_value: float | None = None
    category_id: int | None = None

    @model_validator(mode="after")
    def validate_financial_record(self):
        if self.amount < 0:
            raise ValueError("Amount cannot be negative")
        if self.current_value is not None and self.current_value < 0:
            raise ValueError("Current value cannot be negative")
        if self.target_amount is not None and self.target_amount < self.amount:
            raise ValueError("Goal target amount must be at least the current amount")
        if self.interest_rate is not None and not 0 <= self.interest_rate <= 100:
            raise ValueError("Interest rate must be between 0 and 100")
        if self.frequency is not None and self.frequency.lower() not in {"monthly", "yearly", "annual", "quarterly", "weekly"}:
            raise ValueError("Invalid income frequency")
        return self

class ProfileIn(BaseModel):
    date_of_birth: str | None = None
    city: str | None = None
    risk_profile: str = "moderate"
    retirement_age: int = 60
    annual_rent: float | None = None
    is_hra_claimed: bool = False
    age_band: str = "below_60"


class ProfileUpdate(BaseModel):
    full_name: str | None = Field(default=None, min_length=1, max_length=120)
    date_of_birth: date | None = None
    gender: str | None = Field(default=None, max_length=30)
    city: str | None = Field(default=None, max_length=80)
    state: str | None = Field(default=None, max_length=80)
    occupation: str | None = Field(default=None, max_length=120)
    employment_type: str | None = None
    marital_status: str | None = None
    dependents_count: int | None = Field(default=None, ge=0, le=20)
    risk_profile: str | None = None
    retirement_age: int | None = Field(default=None, ge=18, le=100)
    annual_rent: float | None = Field(default=None, ge=0)
    is_hra_claimed: bool | None = None
    age_band: str | None = None
    health_insurance_cover: bool | None = None
    health_insurance_premium: float | None = Field(default=None, ge=0)
    emergency_fund_target_months: float | None = Field(default=None, ge=0, le=60)
    preferred_currency: str | None = Field(default=None, min_length=3, max_length=3)
    profile_picture: str | None = Field(default=None, max_length=2_000_000)

    @field_validator("risk_profile")
    @classmethod
    def valid_risk_profile(cls, value):
        if value is not None and value not in {"conservative", "moderate", "aggressive"}:
            raise ValueError("Risk profile must be conservative, moderate, or aggressive")
        return value

    @field_validator("date_of_birth")
    @classmethod
    def valid_date_of_birth(cls, value):
        if value is not None and value >= date.today():
            raise ValueError("Date of birth must be in the past")
        return value

    @field_validator("age_band")
    @classmethod
    def valid_age_band(cls, value):
        if value is not None and value not in {"below_60", "60_to_80", "above_80"}:
            raise ValueError("Invalid age band")
        return value


class ProfileCreate(ProfileUpdate):
    full_name: str = Field(min_length=1, max_length=120)


class ProfileResponse(BaseModel):
    full_name: str
    email: str
    date_of_birth: str | None = None
    current_age: int | None = None
    risk_profile: str
    retirement_age: int
    preferred_currency: str = "INR"


class TaxRegimeBreakdownSchema(BaseModel):
    regime: str
    gross_income: float
    taxable_income: float
    total_deductions: float
    final_tax_payable: float


class TaxOptimizationSchema(BaseModel):
    annual_income: float
    comparison: dict
    optimization_suggestions: list[dict]
    disclaimer: str

class RetirementIn(BaseModel):
    current_age: int
    retirement_age: int = 60
    life_expectancy: int = 85
    current_corpus: float = 0
    monthly_sip: float = 0
    sip_step_up: float = 0.10
    annual_expenses: float = 0
    inflation: float = 0.06
    equity_return: float = 0.12
    debt_return: float = 0.07

    @model_validator(mode="after")
    def validate_retirement_plan(self):
        if self.current_age < 0:
            raise ValueError("Current age cannot be negative")
        if self.retirement_age <= self.current_age:
            raise ValueError("Retirement age must be greater than current age")
        if self.life_expectancy < self.retirement_age:
            raise ValueError("Life expectancy must be greater than or equal to retirement age")
        for field_name in ["current_corpus", "monthly_sip", "sip_step_up", "annual_expenses", "inflation", "equity_return", "debt_return"]:
            value = getattr(self, field_name)
            if value < 0:
                raise ValueError(f"{field_name} cannot be negative")
        return self


def serialize(item: Any):
    return {c.name: getattr(item, c.name) for c in item.__table__.columns}

@router.get("/records/{resource}")
def list_records(resource: str, db: Session = Depends(get_db), user: User = Depends(current_user)):
    model = MODELS.get(resource)
    if not model: raise HTTPException(404, "Unknown resource")
    return response([serialize(x) for x in db.query(model).filter_by(user_id=user.id).order_by(model.created_at.desc()).all()])

@router.post("/records/{resource}")
def create_record(resource: str, payload: RecordIn, db: Session = Depends(get_db), user: User = Depends(current_user)):
    model = MODELS.get(resource)
    if not model: raise HTTPException(404, "Unknown resource")
    values = payload.model_dump(exclude_none=True)
    allowed = {c.name for c in model.__table__.columns}
    item = model(user_id=user.id, **{k: v for k, v in values.items() if k in allowed})
    db.add(item); db.commit(); db.refresh(item)
    return response(serialize(item), f"{resource[:-1].capitalize()} added successfully")

@router.put("/records/{resource}/{item_id}")
def update_record(resource: str, item_id: int, payload: RecordIn, db: Session = Depends(get_db), user: User = Depends(current_user)):
    model = MODELS.get(resource)
    item = db.query(model).filter_by(id=item_id, user_id=user.id).first() if model else None
    if not item: raise HTTPException(404, "Record not found")
    allowed = {column.name for column in model.__table__.columns}
    for key, value in payload.model_dump(exclude_none=True).items():
        if key in allowed:
            setattr(item, key, value)
    db.commit(); db.refresh(item)
    return response(serialize(item), f"{resource[:-1].capitalize()} updated successfully")

@router.delete("/records/{resource}/{item_id}")
def delete_record(resource: str, item_id: int, db: Session = Depends(get_db), user: User = Depends(current_user)):
    model = MODELS.get(resource)
    item = db.query(model).filter_by(id=item_id, user_id=user.id).first() if model else None
    if not item: raise HTTPException(404, "Record not found")
    db.delete(item); db.commit()
    return response(None, "Record deleted successfully")

@router.get("/profile/me")
def get_profile(db: Session = Depends(get_db), user: User = Depends(current_user)):
    return response(serialize(db.query(Profile).filter_by(user_id=user.id).first()))

@router.put("/profile/me")
def update_profile(payload: ProfileIn, db: Session = Depends(get_db), user: User = Depends(current_user)):
    profile = db.query(Profile).filter_by(user_id=user.id).first()
    for key, value in payload.model_dump().items(): setattr(profile, key, value)
    db.commit(); db.refresh(profile)
    return response(serialize(profile), "Profile updated successfully")


@router.get("/profile")
def get_current_profile(db: Session = Depends(get_db), user: User = Depends(current_user)):
    return response(get_user_profile(db, user.id), "Profile retrieved successfully")


@router.put("/profile")
def update_current_profile(payload: ProfileUpdate, db: Session = Depends(get_db), user: User = Depends(current_user)):
    try:
        profile = update_user_profile(db, user.id, payload.model_dump(exclude_unset=True))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return response(profile, "Profile updated successfully")

@router.post("/retirement/plan")
def save_retirement(payload: RetirementIn, db: Session = Depends(get_db), user: User = Depends(current_user)):
    plan = db.query(RetirementPlan).filter_by(user_id=user.id).first()
    if not plan: plan = RetirementPlan(user_id=user.id); db.add(plan)
    for key, value in payload.model_dump().items(): setattr(plan, key, value)
    db.commit(); db.refresh(plan)
    return response({"plan": serialize(plan), "projection": project_retirement(**payload.model_dump())}, "Retirement plan saved")

@router.get("/retirement/plan")
def get_retirement(db: Session = Depends(get_db), user: User = Depends(current_user)):
    plan = db.query(RetirementPlan).filter_by(user_id=user.id).first()
    return response({"plan": serialize(plan), "projection": project_retirement(**{c.name: getattr(plan, c.name) for c in RetirementPlan.__table__.columns if c.name not in {"id", "user_id", "created_at", "updated_at"}})} if plan else None)

@router.get("/dashboard")
def dashboard(db: Session = Depends(get_db), user: User = Depends(current_user)):
    incomes = db.query(Income).filter_by(user_id=user.id).all(); expenses = db.query(Expense).filter_by(user_id=user.id).all()
    assets = db.query(Asset).filter_by(user_id=user.id).all(); liabilities = db.query(Liability).filter_by(user_id=user.id).all(); investments = db.query(Investment).filter_by(user_id=user.id).all()
    monthly_income, monthly_expenses = sum(x.amount for x in incomes), sum(x.amount for x in expenses)
    budget = calculate_budget(monthly_income, monthly_expenses)
    wealth = calculate_real_wealth(sum(x.amount for x in assets), sum(x.current_value for x in investments), sum(x.amount for x in liabilities), .06)
    life = calculate_life_value(monthly_income, monthly_expenses, sum(x.amount for x in liabilities), .06)
    score = max(0, min(100, round(50 + budget["savings_rate"] * .7 + (15 if wealth["net_worth"] > 0 else -15))))
    recommendations = []
    if budget["savings_rate"] < 20: recommendations.append({"severity": "warning", "title": "Increase your savings rate", "action": "Aim to save at least 20% of monthly income."})
    if sum(x.amount for x in liabilities) > monthly_income * 6: recommendations.append({"severity": "high", "title": "Review your debt", "action": "Your liabilities are high relative to income."})
    if not recommendations: recommendations.append({"severity": "good", "title": "Healthy foundation", "action": "Keep tracking your financial plan each month."})
    return response({"budget": budget, "wealth": wealth, "life_value": life, "health_score": score, "portfolio": portfolio_summary([serialize(x) for x in investments]), "recommendations": recommendations, "chart": {"income": monthly_income, "expenses": monthly_expenses}, "updated_on": str(date.today())})


@router.get("/predictions/expenses")
def expense_prediction(db: Session = Depends(get_db), user: User = Depends(current_user)):
    expenses = db.query(Expense).filter_by(user_id=user.id).order_by(Expense.created_at.asc()).all()
    prediction = predict_next_expense([expense.amount for expense in expenses])
    return response(prediction, "Expense prediction generated successfully")


@router.get("/recommendations/investments")
def investment_recommendations(db: Session = Depends(get_db), user: User = Depends(current_user)):
    profile = db.query(Profile).filter_by(user_id=user.id).first()
    incomes = db.query(Income).filter_by(user_id=user.id).all()
    expenses = db.query(Expense).filter_by(user_id=user.id).all()
    investments = db.query(Investment).filter_by(user_id=user.id).all()
    budget = calculate_budget(sum(item.amount for item in incomes), sum(item.amount for item in expenses))
    portfolio = portfolio_summary([serialize(item) for item in investments])
    result = recommend_investments(profile.risk_profile if profile else "moderate", budget["savings_rate"], portfolio["allocation"])
    return response(result, "Investment recommendations generated successfully")


@router.get("/tax/optimize")
def tax_optimize(db: Session = Depends(get_db), user: User = Depends(current_user)):
    """JWT-protected tax route; orchestration and calculations are delegated to the service."""
    return response(build_tax_optimization(db, user.id), "Tax optimization generated successfully")
