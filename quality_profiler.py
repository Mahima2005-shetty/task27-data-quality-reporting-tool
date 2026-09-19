import pandas as pd
import numpy as np
import sys
from pathlib import Path


def load_dataset(file_path):
    """Load a CSV dataset."""
    try:
        df = pd.read_csv(file_path)
        print(f"Dataset loaded successfully: {file_path}")
        print(f"Rows: {len(df)}, Columns: {len(df.columns)}")
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)


def profile_dataset(df):
    """Generate column-level data quality metrics."""

    total_rows = len(df)
    report = []

    for column in df.columns:
        missing_count = int(df[column].isna().sum())
        unique_count = int(df[column].nunique(dropna=True))
        duplicate_count = int(df[column].duplicated().sum())

        missing_percentage = (
            (missing_count / total_rows) * 100
            if total_rows else 0
        )

        uniqueness_percentage = (
            (unique_count / total_rows) * 100
            if total_rows else 0
        )

        completeness_percentage = (
            ((total_rows - missing_count) / total_rows) * 100
            if total_rows else 0
        )

        report.append({
            "column": column,
            "data_type": str(df[column].dtype),
            "missing_count": missing_count,
            "missing_percentage": f"{missing_percentage:.2f}%",
            "unique_count": unique_count,
            "uniqueness_percentage": f"{uniqueness_percentage:.2f}%",
            "duplicate_values": duplicate_count,
            "completeness_percentage": f"{completeness_percentage:.2f}%"
        })

    return pd.DataFrame(report)


def detect_invalid_data(df):
    """Detect suspicious and invalid records."""

    invalid = pd.DataFrame(index=df.index)

    # Missing values
    invalid["missing_values"] = df.isna().any(axis=1)

    # Age validation
    if "age" in df.columns:
        age_numeric = pd.to_numeric(
            df["age"],
            errors="coerce"
        )

        invalid["invalid_age"] = (
            df["age"].notna()
            & (
                age_numeric.isna()
                | (age_numeric < 0)
                | (age_numeric > 120)
            )
        )
    else:
        invalid["invalid_age"] = False

    # Salary validation
    if "salary" in df.columns:
        salary_numeric = pd.to_numeric(
            df["salary"],
            errors="coerce"
        )

        invalid["invalid_salary"] = (
            df["salary"].notna()
            & salary_numeric.isna()
        )
    else:
        invalid["invalid_salary"] = False

    # Email validation
    if "email" in df.columns:
        email_pattern = r"^[^@\s]+@[^@\s]+\.[^@\s]+$"

        invalid["invalid_email"] = (
            df["email"].notna()
            & ~df["email"].astype(str).str.match(
                email_pattern,
                na=False
            )
        )
    else:
        invalid["invalid_email"] = False

    # Duplicate detection ignoring ID columns
    duplicate_columns = [
        column for column in df.columns
        if column.lower() not in ["id", "employee_id"]
    ]

    if duplicate_columns:
        invalid["duplicate_record"] = df.duplicated(
            subset=duplicate_columns,
            keep=False
        )
    else:
        invalid["duplicate_record"] = df.duplicated(
            keep=False
        )

    invalid["is_suspicious"] = invalid.any(axis=1)

    return invalid


def create_invalid_record_report(df, invalid):
    """Create a report containing suspicious records."""

    mask = invalid["is_suspicious"]

    invalid_records = df[mask].copy()

    issue_columns = [
        column
        for column in invalid.columns
        if column != "is_suspicious"
    ]

    invalid_records["quality_issues"] = invalid.loc[
        mask,
        issue_columns
    ].apply(
        lambda row: ", ".join(
            column.replace("_", " ").title()
            for column in issue_columns
            if row[column]
        ),
        axis=1
    )

    return invalid_records


def save_reports(quality_report, invalid_records):
    """Save reports to the reports folder."""

    reports_folder = Path("reports")
    reports_folder.mkdir(exist_ok=True)

    quality_report.to_csv(
        reports_folder / "data_quality_report.csv",
        index=False
    )

    invalid_records.to_csv(
        reports_folder / "invalid_records.csv",
        index=False
    )

    print("\nReports generated successfully!")
    print(
        "Quality report: "
        "reports\\data_quality_report.csv"
    )
    print(
        "Invalid records: "
        "reports\\invalid_records.csv"
    )


def print_summary(df, quality_report, invalid_records):
    """Display overall data quality metrics."""

    total_cells = df.shape[0] * df.shape[1]
    missing_cells = int(df.isna().sum().sum())

    duplicate_columns = [
        column for column in df.columns
        if column.lower() not in ["id", "employee_id"]
    ]

    if duplicate_columns:
        duplicate_rows = int(
            df.duplicated(
                subset=duplicate_columns
            ).sum()
        )
    else:
        duplicate_rows = int(
            df.duplicated().sum()
        )

    suspicious_rows = len(invalid_records)

    completeness = (
        ((total_cells - missing_cells) / total_cells) * 100
        if total_cells else 0
    )

    # Convert percentage strings back to numbers
    uniqueness_values = (
        quality_report["uniqueness_percentage"]
        .str.rstrip("%")
        .astype(float)
    )

    average_uniqueness = uniqueness_values.mean()

    print("\n" + "=" * 60)
    print("DATA QUALITY SUMMARY")
    print("=" * 60)

    print(f"Total records       : {len(df)}")
    print(f"Total columns       : {len(df.columns)}")
    print(f"Missing cells       : {missing_cells}")
    print(f"Duplicate records   : {duplicate_rows}")
    print(f"Suspicious records  : {suspicious_rows}")
    print(f"Completeness        : {completeness:.2f}%")
    print(f"Average uniqueness  : {average_uniqueness:.2f}%")

    print("\nColumn Quality Report:")
    print(quality_report.to_string(index=False))

    print("\n" + "=" * 60)


def main():
    """Main program."""

    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = "data/sample_data.csv"

    print("=" * 60)
    print("DATA QUALITY REPORTING TOOL")
    print("=" * 60)

    df = load_dataset(file_path)

    quality_report = profile_dataset(df)

    invalid = detect_invalid_data(df)

    invalid_records = create_invalid_record_report(
        df,
        invalid
    )

    save_reports(
        quality_report,
        invalid_records
    )

    print_summary(
        df,
        quality_report,
        invalid_records
    )


if __name__ == "__main__":
    main()