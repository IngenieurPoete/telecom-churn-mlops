"""
validate_data.py - Compatible Great Expectations 1.17.x
"""
import great_expectations as gx
from great_expectations.expectations import (
    ExpectColumnValuesToNotBeNull,
    ExpectColumnValuesToBeBetween,
    ExpectColumnValuesToBeInSet,
    ExpectTableRowCountToBeBetween,
    ExpectTableColumnsToMatchSet,
)
import pandas as pd


def validate_dataset(filepath: str) -> bool:

    df = pd.read_csv(filepath)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    context = gx.get_context(mode="ephemeral")

    # ── Source de données ──────────────────────────────────────────
    datasource     = context.data_sources.add_pandas("pandas_source")
    asset          = datasource.add_dataframe_asset("churn_asset")
    batch_def      = asset.add_batch_definition_whole_dataframe("full_batch")

    # ── Suite d'expectations ───────────────────────────────────────
    suite = context.suites.add(gx.ExpectationSuite(name="churn_suite"))

    suite.add_expectation(ExpectTableColumnsToMatchSet(
        column_set=["tenure", "MonthlyCharges", "TotalCharges",
                    "Churn", "Contract", "InternetService"],
        exact_match=False
    ))
    suite.add_expectation(
        ExpectColumnValuesToNotBeNull(column="tenure"))
    suite.add_expectation(
        ExpectColumnValuesToNotBeNull(column="MonthlyCharges"))
    suite.add_expectation(
        ExpectColumnValuesToNotBeNull(column="Churn"))
    suite.add_expectation(
        ExpectColumnValuesToBeBetween(
            column="tenure", min_value=0, max_value=72))
    suite.add_expectation(
        ExpectColumnValuesToBeBetween(
            column="MonthlyCharges", min_value=0, max_value=200))
    suite.add_expectation(
        ExpectColumnValuesToBeInSet(
            column="Churn", value_set=["Yes", "No"]))
    suite.add_expectation(
        ExpectTableRowCountToBeBetween(
            min_value=1000, max_value=100000))

    # ── Validation ─────────────────────────────────────────────────
    validation_def = context.validation_definitions.add(
        gx.ValidationDefinition(
            name="churn_validation",
            data=batch_def,
            suite=suite,
        )
    )

    results = validation_def.run(
        batch_parameters={"dataframe": df}
    )

    # ── Affichage ──────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  GREAT EXPECTATIONS — RAPPORT DE VALIDATION")
    print("=" * 55)

    passed = 0
    failed = 0
    for res in results.results:
        status  = "✅ PASS" if res.success else "❌ FAIL"
        exptype = res.expectation_config.type
        print(f"  {status}  →  {exptype}")
        if res.success:
            passed += 1
        else:
            failed += 1

    print("-" * 55)
    print(f"  Réussis : {passed} | Échoués : {failed}")
    print(f"  Résultat global : {'✅ VALIDATION RÉUSSIE' if results.success else '❌ VALIDATION ÉCHOUÉE'}")
    print("=" * 55)

    return results.success


if __name__ == "__main__":
    ok = validate_dataset(
        "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    )
    exit(0 if ok else 1)