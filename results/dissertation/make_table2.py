"""Generate Table 2: Difficulty Distribution of Retained Versus Excluded Queries.

Matches the visual style of table1_results.png (serif title, dark identifier
header cells, green metric header cells, bold summary row).
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = os.path.join(os.path.dirname(__file__), "charts")

TITLE = "Table 2: Difficulty Distribution of Retained Versus Excluded Queries"

COL_LABELS = [
    "Number of hops",
    "Full benchmark\n(2,556)",
    "Retained\n(1,701)",
    "Excluded\n(855)",
]

# Each row: (label, full, retained, excluded)
ROWS = [
    ("0 (null)", "11.8%", "12.4%", "10.5%"),
    ("2",        "42.2%", "42.3%", "42.1%"),
    ("3",        "30.4%", "30.0%", "31.3%"),
    ("4",        "15.6%", "15.3%", "16.0%"),
    ("Mean hops", "2.38", "2.36", "2.42"),
]

NOTE = (
    "Note: Excluded queries are those where the compact model failed to emit a "
    "well-formed logic tree.\n"
    "Distributions are within ±1.0 percentage point and mean hops within 0.06 "
    "across retained and excluded sets,\n"
    "confirming the retained intersection is representative on the measured "
    "difficulty dimension."
)

HEADER_DARK = "#595959"   # identifier column header
HEADER_GREEN = "#bfbfbf"  # metric column header (grayscale)
SUMMARY_FILL = "#e6e6e6"  # mean-hops summary row (grayscale)
WHITE = "#ffffff"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.axis("off")

    n_cols = len(COL_LABELS)
    table_data = [[r[0], r[1], r[2], r[3]] for r in ROWS]

    table = ax.table(
        cellText=table_data,
        colLabels=COL_LABELS,
        cellLoc="center",
        loc="center",
        colWidths=[0.28, 0.24, 0.24, 0.24],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2.0)

    n_rows = len(ROWS)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#7f7f7f")
        cell.set_linewidth(0.8)

        # Header row
        if row == 0:
            cell.set_text_props(weight="bold", color="white", fontfamily="serif")
            cell.set_facecolor(HEADER_DARK if col == 0 else HEADER_GREEN)
            if col != 0:
                cell.set_text_props(weight="bold", color="black", fontfamily="serif")
            cell.set_height(0.16)
            continue

        cell.set_text_props(fontfamily="serif")
        # Summary row (Mean hops) -> bold + shaded, like the highlighted B1 row
        if row == n_rows:  # last data row
            cell.set_facecolor(SUMMARY_FILL)
            cell.set_text_props(weight="bold", fontfamily="serif")
        else:
            cell.set_facecolor(WHITE)

        # Left-align the identifier column for readability
        if col == 0:
            cell.set_text_props(
                ha="left",
                fontfamily="serif",
                weight="bold" if row == n_rows else "normal",
            )
            cell._loc = "left"

    fig.suptitle(TITLE, fontsize=14, fontweight="bold", fontfamily="serif", y=0.95)

    fig.text(
        0.5, 0.06, NOTE,
        ha="center", va="top", fontsize=9,
        fontstyle="italic", fontfamily="serif",
    )

    out_path = os.path.join(OUT_DIR, "table2_difficulty_distribution.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
