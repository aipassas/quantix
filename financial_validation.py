"""Financial statement field validation for Quantix.

Yahoo Finance frequently renames or omits financial statement line items.
Reading a missing field with a bare `.loc[...]` raises a KeyError that
calculation code has historically swallowed with a blanket `except: pass`,
which hides exactly which field was missing and silently produces
misleading numbers (e.g. treating a missing Total Debt as $0).

This module gives calculation code two things instead:
  1. `get_field()` — a safe, alias-aware accessor that returns None (never
     raises) when a field is genuinely absent or null, so callers can show
     "N/A" instead of computing with a fabricated default.
  2. `validate_financials()` — a structural check of which required/optional
     fields are present per statement, used to render an explicit
     validation report instead of failing silently.
"""
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Tuple

import pandas as pd

# (canonical display name, accepted field-name aliases in priority order, required)
BALANCE_SHEET_FIELDS: Tuple[Tuple[str, Tuple[str, ...], bool], ...] = (
    ("Total Assets", ("Total Assets",), True),
    ("Current Assets", ("Current Assets",), True),
    ("Current Liabilities", ("Current Liabilities",), True),
    ("Stockholders Equity", ("Stockholders Equity",), True),
    ("Total Liabilities (Net Minority Interest)", ("Total Liabilities Net Minority Interest",), True),
    ("Total Debt", ("Total Debt",), False),
    ("Retained Earnings", ("Retained Earnings",), False),
    # Optional, not just required-with-a-lower-bar: many companies (banks,
    # service/software businesses) genuinely carry no inventory, so its
    # absence is structural rather than a data gap.
    ("Inventory", ("Inventory",), False),
    # For Enterprise Value (EV = Market Cap + Total Debt − Cash). The
    # narrower line item is preferred; the broader aggregate (which also
    # folds in short-term investments) is an acceptable fallback.
    ("Cash And Cash Equivalents", ("Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments"), False),
)

INCOME_STATEMENT_FIELDS: Tuple[Tuple[str, Tuple[str, ...], bool], ...] = (
    ("Total Revenue", ("Total Revenue",), True),
    ("EBIT (or Operating Income)", ("EBIT", "Operating Income"), True),
    ("Net Income", ("Net Income",), True),
    ("Interest Expense", ("Interest Expense",), False),
    # Optional, not just required-with-a-lower-bar: banks and other financials
    # don't report cost of revenue at all, so Gross Profit is structurally
    # absent rather than merely missing. Same for the effective-tax-rate pair.
    ("Gross Profit", ("Gross Profit",), False),
    ("Operating Income", ("Operating Income",), False),
    ("Pretax Income", ("Pretax Income",), False),
    ("Tax Provision", ("Tax Provision",), False),
)

CASH_FLOW_FIELDS: Tuple[Tuple[str, Tuple[str, ...], bool], ...] = (
    ("Free Cash Flow", ("Free Cash Flow",), True),
    # For EBITDA = EBIT + D&A. Optional: not every filer breaks this out
    # separately from other reconciling items.
    ("Depreciation And Amortization", ("Depreciation And Amortization", "Depreciation Amortization Depletion"), False),
)


def get_field(df: Optional[pd.DataFrame], aliases: Sequence[str], default: Any = None) -> Any:
    """Return the most recent-period value for the first matching alias in `aliases`.

    Never raises: returns `default` if the statement is missing/empty, no
    alias is present in its index, or the matched value is null.
    """
    if df is None or df.empty:
        return default
    for alias in aliases:
        if alias in df.index:
            try:
                value = df.loc[alias].iloc[0]
            except Exception:
                continue
            if pd.notna(value):
                return value
    return default


@dataclass
class FieldCheck:
    name: str
    required: bool
    present: bool


@dataclass
class StatementValidation:
    statement_name: str
    checks: List[FieldCheck] = field(default_factory=list)

    @property
    def missing_required(self) -> List[str]:
        return [c.name for c in self.checks if c.required and not c.present]

    @property
    def missing_optional(self) -> List[str]:
        return [c.name for c in self.checks if not c.required and not c.present]

    @property
    def is_valid(self) -> bool:
        """False when a REQUIRED field is missing. Missing optional fields don't invalidate the statement."""
        return not self.missing_required

    @property
    def warnings(self) -> List[str]:
        messages = [f"{self.statement_name}: missing required field '{name}'." for name in self.missing_required]
        messages += [f"{self.statement_name}: optional field '{name}' not reported." for name in self.missing_optional]
        return messages


def _validate_statement(df: Optional[pd.DataFrame], specs: Tuple[Tuple[str, Tuple[str, ...], bool], ...], statement_name: str) -> StatementValidation:
    validation = StatementValidation(statement_name=statement_name)
    for name, aliases, required in specs:
        present = get_field(df, aliases) is not None
        validation.checks.append(FieldCheck(name=name, required=required, present=present))
    return validation


def validate_balance_sheet(df: Optional[pd.DataFrame]) -> StatementValidation:
    return _validate_statement(df, BALANCE_SHEET_FIELDS, "Balance Sheet")


def validate_income_statement(df: Optional[pd.DataFrame]) -> StatementValidation:
    return _validate_statement(df, INCOME_STATEMENT_FIELDS, "Income Statement")


def validate_cash_flow(df: Optional[pd.DataFrame]) -> StatementValidation:
    return _validate_statement(df, CASH_FLOW_FIELDS, "Cash Flow Statement")


@dataclass
class ValidationReport:
    balance_sheet: StatementValidation
    income_statement: StatementValidation
    cash_flow: StatementValidation

    @property
    def is_valid(self) -> bool:
        return self.balance_sheet.is_valid and self.income_statement.is_valid and self.cash_flow.is_valid

    @property
    def statements(self) -> List[StatementValidation]:
        return [self.balance_sheet, self.income_statement, self.cash_flow]

    @property
    def all_warnings(self) -> List[str]:
        return self.balance_sheet.warnings + self.income_statement.warnings + self.cash_flow.warnings


def validate_financials(income_stmt: Optional[pd.DataFrame], balance_sheet: Optional[pd.DataFrame], cash_flow: Optional[pd.DataFrame]) -> ValidationReport:
    """Validate all three financial statements at once and return a combined report."""
    return ValidationReport(
        balance_sheet=validate_balance_sheet(balance_sheet),
        income_statement=validate_income_statement(income_stmt),
        cash_flow=validate_cash_flow(cash_flow),
    )
