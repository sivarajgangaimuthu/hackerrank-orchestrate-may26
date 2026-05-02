"""CLI entry point for generating support ticket predictions."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from agent import TicketTriageAgent
from ticket_logging import append_ticket_log, ensure_log_file, log_path


OUTPUT_COLUMNS = ("status", "product_area", "response", "justification", "request_type")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _clean_cell(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"none", "null", "nan"}:
        return ""
    return text


def _get_case_insensitive(row: dict[str, Any], column_name: str) -> str:
    target = column_name.lower()
    for key, value in row.items():
        if key and key.lower() == target:
            return _clean_cell(value)
    return ""


def read_tickets(input_path: Path) -> list[dict[str, str]]:
    """Read support tickets while tolerating missing/null values."""

    with input_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        tickets: list[dict[str, str]] = []
        for row in reader:
            tickets.append(
                {
                    "issue": _get_case_insensitive(row, "issue"),
                    "subject": _get_case_insensitive(row, "subject"),
                    "company": _get_case_insensitive(row, "company"),
                }
            )
    return tickets


def write_outputs(output_path: Path, rows: list[dict[str, str]]) -> None:
    """Write evaluator output columns in deterministic order."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in OUTPUT_COLUMNS})


def process_tickets(input_path: Path, output_path: Path, use_gemini: bool = True) -> dict[str, int]:
    tickets = read_tickets(input_path)
    total = len(tickets)
    ticket_log_path = ensure_log_file()

    print(f"Loading support corpus and building retrieval index...")
    agent = TicketTriageAgent(use_gemini=use_gemini)
    print(f"Processing {total} ticket(s) from {input_path}")
    print(f"Appending ticket logs to {ticket_log_path}")

    outputs: list[dict[str, str]] = []
    replied_count = 0
    escalated_count = 0

    for index, ticket in enumerate(tickets, start=1):
        issue = ticket["issue"]
        subject = ticket["subject"]
        company = ticket["company"] or None

        result = agent.process_ticket(issue, subject=subject, company=company)
        outputs.append(result)
        append_ticket_log(
            subject=subject,
            company=company,
            issue=issue,
            request_type=result["request_type"],
            product_area=result["product_area"],
            status=result["status"],
            response=result["response"],
            path=ticket_log_path,
        )

        if result["status"] == "replied":
            replied_count += 1
        elif result["status"] == "escalated":
            escalated_count += 1

        print(
            f"[{index}/{total}] status={result['status']} "
            f"type={result['request_type']} area={result['product_area']}"
        )

    write_outputs(output_path, outputs)
    print(f"Wrote predictions to {output_path}")
    print("Summary:")
    print(f"  total tickets: {total}")
    print(f"  replied: {replied_count}")
    print(f"  escalated: {escalated_count}")

    return {
        "total": total,
        "replied": replied_count,
        "escalated": escalated_count,
    }


def parse_args() -> argparse.Namespace:
    root = _repo_root()
    parser = argparse.ArgumentParser(description="Run the support triage agent over a CSV file.")
    parser.add_argument(
        "--input",
        type=Path,
        default=root / "support_tickets" / "support_tickets.csv",
        help="Input support tickets CSV path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "support_tickets" / "output.csv",
        help="Output predictions CSV path.",
    )
    parser.add_argument(
        "--no-gemini",
        action="store_true",
        help="Disable Gemini calls and use deterministic local fallback responses.",
    )
    parser.epilog = f"Ticket processing logs are appended to: {log_path()}"
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input}")

    process_tickets(
        input_path=args.input,
        output_path=args.output,
        use_gemini=not args.no_gemini,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
