"""Generate Tables F1 (coding rubric) and F2 (coded sample) — grayscale only.

Matches the house style: serif type, dark-gray identifier header cells,
medium-gray header cells, white body rows. These tables carry long prose cells,
so text is wrapped and row heights are sized to the wrapped line count.
"""
import os
import textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox

OUT_DIR = os.path.join(os.path.dirname(__file__), "charts")

HEADER_DARK = "#595959"   # identifier-column header (white text)
HEADER_GRAY = "#bfbfbf"   # other header cells (black text)
PARTIAL_FILL = "#e6e6e6"  # "Partial" verdict shading
FAIL_FILL = "#bdbdbd"     # "Fail" verdict shading
WHITE = "#ffffff"
EDGE = "#7f7f7f"
LINE_H = 0.052            # axes-fraction height per wrapped text line


def _wrap(text, width):
    return textwrap.fill(text, width)


def _render(title, cols, col_widths, wrap_widths, rows, left_cols,
            ident_cols, verdict_cols, out_name, figsize, note=None,
            header_h_lines=2, loc="upper center"):
    """Render a text table with per-row heights sized to wrapped content."""
    # Wrap every body cell and remember its line count
    wrapped, line_counts = [], []
    for row in rows:
        wrow, max_lines = [], 1
        for j, val in enumerate(row):
            w = _wrap(val, wrap_widths[j]) if wrap_widths[j] else val
            wrow.append(w)
            max_lines = max(max_lines, w.count("\n") + 1)
        wrapped.append(wrow)
        line_counts.append(max_lines)

    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    table = ax.table(cellText=wrapped, colLabels=cols, cellLoc="center",
                     loc=loc, colWidths=col_widths)
    table.auto_set_font_size(False)
    table.set_fontsize(10)

    n_rows = len(rows)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor(EDGE)
        cell.set_linewidth(0.8)
        cell.set_text_props(fontfamily="serif")
        if r == 0:
            cell.set_height(LINE_H * header_h_lines + 0.012)
            cell.set_facecolor(HEADER_DARK if c in ident_cols else HEADER_GRAY)
            cell.set_text_props(weight="bold", fontfamily="serif",
                                color="white" if c in ident_cols else "black")
            if c in left_cols:
                cell.set_text_props(weight="bold", ha="left", fontfamily="serif",
                                    color="white" if c in ident_cols else "black")
                cell._loc = "left"
            continue

        cell.set_height(LINE_H * line_counts[r - 1] + 0.012)
        # Verdict shading (grayscale)
        fill = WHITE
        if c in verdict_cols:
            v = rows[r - 1][c].strip().lower()
            if v == "partial":
                fill = PARTIAL_FILL
            elif v == "fail":
                fill = FAIL_FILL
        cell.set_facecolor(fill)
        if c in ident_cols:
            cell.set_text_props(weight="bold", fontfamily="serif")
        if c in left_cols:
            cell.set_text_props(ha="left", fontfamily="serif",
                                weight="bold" if c in ident_cols else "normal")
            cell._loc = "left"

    fig.suptitle(title, fontsize=13.5, fontweight="bold", fontfamily="serif",
                 y=0.98)
    note_artist = None
    if note:
        note_artist = fig.text(0.5, 0.04, _wrap(note, 130), ha="center",
                               va="top", fontsize=8.5, fontstyle="italic",
                               fontfamily="serif")

    # Crop tightly to the drawn artists (table + title + note), avoiding the
    # empty axes area left over when the figure is taller than the content.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    artists = [table, fig._suptitle]
    if note_artist is not None:
        artists.append(note_artist)
    bb = Bbox.union([a.get_window_extent(renderer) for a in artists])
    bb = bb.transformed(fig.dpi_scale_trans.inverted())
    bb = bb.expanded(1.0, 1.0).padded(0.12)

    out = os.path.join(OUT_DIR, out_name)
    fig.savefig(out, dpi=150, bbox_inches=bb, facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")


def table_e1():
    title = "Table E1: Coding Rubric for Logic-Tree Quality"
    cols = ["Criterion", "Definition", "Scoring"]
    rows = [
        ["Premise validity",
         "Whether each Fact From Story node stated something the retrieved "
         "passage actually supported, rather than a misreading or fabrication",
         "Pass / Partial / Fail, scored per tree on its grounding nodes"],
        ["Entailment correctness",
         "Whether the deduction each parent node drew from its children actually "
         "followed from them",
         "Pass / Partial / Fail"],
        ["Grounding completeness",
         "Whether the Commonsense Knowledge nodes supplying the warrants were "
         "necessary and defensible rather than vacuous padding",
         "Pass / Partial / Fail"],
    ]
    _render(
        title=title,
        cols=cols,
        col_widths=[0.20, 0.56, 0.24],
        wrap_widths=[24, 58, 28],
        rows=rows,
        left_cols={1},
        ident_cols={0},
        verdict_cols=set(),
        out_name="tableE1_coding_rubric.png",
        figsize=(13, 3.4),
        header_h_lines=1,
    )


def table_e2():
    title = "Table E2: Coded Sample of Logic Trees (Appendix D)"
    cols = ["Tree (Appendix D)", "Premise\nvalidity", "Entailment\ncorrectness",
            "Grounding\ncompleteness", "Note"]
    rows = [
        ["Ex. 1 — FTX / Bankman-Fried", "Pass", "Pass", "Pass",
         "Facts quoted accurately from indictment passages; conclusion follows; "
         "warrants (“fraud is a crime”) defensible."],
        ["Ex. 2 — AfroFuture diversity", "Pass", "Pass", "Partial",
         "Facts grounded; entailment sound; one warrant (“if people from "
         "different countries attend, the event is global”) borders on "
         "restating the conclusion."],
        ["Ex. 3 — Britney Spears conservatorship", "Pass", "Pass", "Pass",
         "Single grounding fact accurate; conservatorship warrant carries genuine "
         "inferential weight."],
        ["Ex. 4 — Man United / Man City", "Pass", "Partial", "Pass",
         "Facts grounded; the tree leans on a draw-result fact to support a "
         "commercial-operation claim, a partial entailment gap."],
        ["Ex. 5 — Apple Watch smartwatch", "Partial", "Fail", "Partial",
         "Facts about price are accurate, but the tree infers a general claim "
         "about smartwatch functionality from single-product price facts — an "
         "entailment failure, consistent with the Appendix C note."],
    ]
    note = ("Note. Shading: light gray = Partial, medium gray = Fail.")
    _render(
        title=title,
        cols=cols,
        col_widths=[0.22, 0.11, 0.12, 0.13, 0.42],
        wrap_widths=[26, 0, 0, 0, 52],
        rows=rows,
        left_cols={4},
        ident_cols={0},
        verdict_cols={1, 2, 3},
        out_name="tableE2_coded_sample.png",
        figsize=(14, 4.6),
        note=note,
        header_h_lines=2,
    )


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    table_e1()
    table_e2()
