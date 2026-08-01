"""Generate appendix Tables A1-A3 (grayscale only).

Matches the visual style of table1_results.png / table2: serif type, dark-gray
identifier header cells, medium-gray metric header cells, white body rows, and a
light-gray shaded bold row for the highlighted (B1) configuration.
"""
import os
import textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = os.path.join(os.path.dirname(__file__), "charts")

# Grayscale palette (no color)
HEADER_DARK = "#595959"   # identifier-column header (white text)
HEADER_GRAY = "#bfbfbf"   # metric-column header (black text)
HIGHLIGHT = "#e6e6e6"     # shaded highlighted row
DIAG = "#d9d9d9"          # diagonal cells in the head-to-head matrix
WHITE = "#ffffff"
EDGE = "#7f7f7f"


def _wrap_note(text, width=120):
    return "\n".join(textwrap.fill(p, width) for p in text.split("\n"))


def _style_common(table):
    table.auto_set_font_size(False)
    for cell in table.get_celld().values():
        cell.set_edgecolor(EDGE)
        cell.set_linewidth(0.8)
        cell.set_text_props(fontfamily="serif")


# ---------------------------------------------------------------------------
# Table A1 — full experimental results
# ---------------------------------------------------------------------------
def table_a1():
    title = ("Table A1: Complete Experimental Results Across All Seven "
             "Configurations (n = 1,701)")
    cols = ["Config", "Model", "Tier", "Pipeline", "Accuracy\n(EM)", "Recall",
            "Gen.\ncost", "Tree\ncost", "Total\ncost", "Cost/\nquery",
            "Avg latency\n(s)"]
    n_ident = 4  # first four columns are identifiers (dark header)
    rows = [
        ["A1", "Llama-3.1-8B", "Cheap", "RAG only", "59.9%", "0.448", "$0.323", "$0.000", "$0.323", "$0.00019", "4.9"],
        ["A2", "Claude 3 Haiku", "Cheap", "RAG only", "59.4%", "0.440", "$2.803", "$0.000", "$2.803", "$0.00165", "2.3"],
        ["A4", "Claude Sonnet 4.6", "Mid", "RAG only", "65.3%", "0.500", "$10.206", "$0.000", "$10.206", "$0.00600", "5.5"],
        ["A3", "GPT-5.4", "Expensive", "RAG only", "78.3%", "0.618", "$14.390", "$0.000", "$14.390", "$0.00846", "1.9"],
        ["B1", "Llama-3.1-8B", "Cheap", "RAG + tree", "63.9%", "0.498", "$0.342", "$0.851", "$1.193", "$0.00070", "16.4"],
        ["C1", "Llama-3.1-8B", "Cheap", "Tree only", "38.6%", "0.240", "$0.035", "$0.851", "$0.886", "$0.00052", "1.5"],
        ["E1", "Llama-3.1-8B", "Cheap", "RAG + fake tree", "62.7%", "0.482", "$0.350", "$0.000", "$0.350", "$0.00021", "2.8"],
    ]
    highlight = 5  # B1 (1-indexed table row incl. header)
    note = ("Note. EM = Exact Match. Costs are totals in USD across the 1,701-query "
            "evaluation set; cost/query is the per-query mean. B1 adds the external "
            "reasoning engine to the A1 baseline; C1 and E1 are ablations isolating, "
            "respectively, the tree's content without retrieved passages and the "
            "prompt-scaffolding effect of a non-informative placeholder tree.")

    widths = [0.055, 0.135, 0.075, 0.115, 0.085, 0.07, 0.075, 0.075, 0.075, 0.08, 0.105]
    fig, ax = plt.subplots(figsize=(16, 4.0))
    ax.axis("off")
    ax.set_position([0.02, 0.12, 0.96, 0.74])
    table = ax.table(cellText=rows, colLabels=cols, cellLoc="center",
                     loc="center", colWidths=widths)
    table.set_fontsize(10.5)
    table.scale(1, 2.4)
    _style_common(table)

    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor(HEADER_DARK if c < n_ident else HEADER_GRAY)
            cell.set_text_props(weight="bold", fontfamily="serif",
                                color="white" if c < n_ident else "black")
            cell.set_height(0.20)
        else:
            cell.set_facecolor(HIGHLIGHT if r == highlight else WHITE)
            cell.set_text_props(weight="bold" if r == highlight else "normal",
                                fontfamily="serif")

    fig.suptitle(title, fontsize=13.5, fontweight="bold", fontfamily="serif", y=0.97)
    fig.text(0.5, 0.05, _wrap_note(note, 150), ha="center", va="top",
             fontsize=8.5, fontstyle="italic", fontfamily="serif")
    out = os.path.join(OUT_DIR, "tableA1_full_results.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Table A2 — head-to-head win counts (row beats column)
# ---------------------------------------------------------------------------
def table_a2():
    title = "Table A2: Head-to-Head Win Counts (n = 1,701)"
    subtitle = ("Each cell counts the queries the row configuration answered "
                "correctly while the column configuration did not.")
    labels = ["A1", "A2", "A3", "A4", "B1", "C1", "E1"]
    # matrix[row][col]; diagonal = None
    matrix = {
        "A1": [None, 123, 52, 86, 70, 413, 88],
        "A2": [115, None, 48, 55, 93, 392, 101],
        "A3": [365, 369, None, 275, 293, 715, 315],
        "A4": [177, 154, 53, None, 141, 486, 154],
        "B1": [138, 169, 48, 118, None, 477, 108],
        "C1": [51, 38, 40, 33, 47, None, 44],
        "E1": [136, 157, 50, 111, 88, 454, None],
    }
    cols = ["Config"] + labels
    rows = []
    for rl in labels:
        cells = ["—" if v is None else str(v) for v in matrix[rl]]
        rows.append([rl] + cells)

    note = ("Note. Read row-versus-column. B1 beat its own A1 baseline on 138 "
            "queries while losing on 70, a net advantage of 68 queries from adding "
            "the reasoning engine. GPT-5.4 (A3) dominated every configuration.")

    widths = [0.13] + [0.124] * 7
    fig, ax = plt.subplots(figsize=(11.5, 4.2))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=cols, cellLoc="center",
                     loc="center", colWidths=widths)
    table.set_fontsize(11)
    table.scale(1, 2.0)
    _style_common(table)

    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor(HEADER_DARK if c == 0 else HEADER_GRAY)
            cell.set_text_props(weight="bold", fontfamily="serif",
                                color="white" if c == 0 else "black")
            cell.set_height(0.13)
        elif c == 0:
            cell.set_facecolor(HEADER_DARK)
            cell.set_text_props(weight="bold", color="white", fontfamily="serif")
        elif r == c:  # diagonal (row index r maps to col index c)
            cell.set_facecolor(DIAG)
        else:
            cell.set_facecolor(WHITE)

    fig.suptitle(title, fontsize=14, fontweight="bold", fontfamily="serif", y=1.0)
    fig.text(0.5, 0.9, subtitle, ha="center", va="top", fontsize=9.5,
             fontstyle="italic", fontfamily="serif")
    fig.text(0.5, 0.03, _wrap_note(note, 120), ha="center", va="top",
             fontsize=9, fontstyle="italic", fontfamily="serif")
    out = os.path.join(OUT_DIR, "tableA2_head_to_head.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Table A3 — accuracy by question type
# ---------------------------------------------------------------------------
def table_a3():
    title = "Table A3: Accuracy by Question Type (Exact Match %, n = 1,701)"
    cols = ["Config", "Inference\n(n = 541)", "Comparison\n(n = 568)",
            "Temporal\n(n = 381)", "Null\n(n = 211)", "Overall"]
    rows = [
        ["A1", "88.2", "53.5", "62.5", "0.0", "59.9"],
        ["A2", "91.9", "48.2", "63.0", "0.0", "59.4"],
        ["A3", "95.4", "83.3", "90.0", "0.0", "78.3"],
        ["A4", "93.9", "62.9", "64.3", "0.0", "65.3"],
        ["B1", "91.5", "58.5", "68.2", "0.0", "63.9"],
        ["C1", "50.5", "35.0", "48.6", "0.0", "38.6"],
        ["E1", "91.3", "58.1", "63.8", "0.0", "62.7"],
    ]
    highlight = 5  # B1
    note = ("Note. Null queries, for which no answer exists in the corpus, scored "
            "0.0% across all configurations because the Exact Match scorer compared "
            "generated text against a gold short answer and could not credit a "
            "correct abstention; this affected every configuration identically and "
            "therefore did not bias the relative comparison. Against the A1 baseline, "
            "adding the reasoning engine (B1) improved comparison accuracy by 5.0 "
            "points and temporal accuracy by 5.7 points, the multi-step types where "
            "structured reasoning was expected to help most.")

    widths = [0.13, 0.175, 0.18, 0.165, 0.135, 0.135]
    fig, ax = plt.subplots(figsize=(12, 3.8))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=cols, cellLoc="center",
                     loc="center", colWidths=widths)
    table.set_fontsize(11)
    table.scale(1, 2.2)
    _style_common(table)

    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor(HEADER_DARK if c == 0 else HEADER_GRAY)
            cell.set_text_props(weight="bold", fontfamily="serif",
                                color="white" if c == 0 else "black")
            cell.set_height(0.18)
        else:
            cell.set_facecolor(HIGHLIGHT if r == highlight else WHITE)
            cell.set_text_props(weight="bold" if r == highlight else "normal",
                                fontfamily="serif")

    fig.suptitle(title, fontsize=14, fontweight="bold", fontfamily="serif", y=1.0)
    fig.text(0.5, 0.02, _wrap_note(note, 135), ha="center", va="top",
             fontsize=8.5, fontstyle="italic", fontfamily="serif")
    out = os.path.join(OUT_DIR, "tableA3_accuracy_by_type.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    table_a1()
    table_a2()
    table_a3()
