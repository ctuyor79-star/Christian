"""Task 10 — use-case diagram (matplotlib preview + draw.io XML)."""
import io
import textwrap
from xml.sax.saxutils import quoteattr

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


BOUNDARY = ("Campus Facility Reservation and Monitoring System", 190, 30, 670, 600)
ACTORS = {
    "actor_req": ("Requester\n(Student / Faculty / Staff)", 70, 215),
    "actor_adm": ("Facility\nAdministrator", 920, 270),
}
USE_CASES = {
    "uc_submit":  ("Submit Reservation Request",            215, 100, False),
    "uc_avail":   ("Check Facility Availability",           215, 240, False),
    "uc_cancel":  ("Cancel Reservation",                    215, 380, False),
    "uc_review":  ("Review Reservation Request",            655, 100, False),
    "uc_record":  ("Record Facility Usage",                 655, 240, False),
    "uc_dash":    ("View Analytics Dashboard",              655, 380, True),
    "uc_risk":    ("View Cancellation Risk Prediction",     435, 190, True),
    "uc_explain": ("Show Risk Explanation",                 435, 330, True),
}
UC_W, UC_H = 180, 70
LINKS = [
    ("a1", "actor_req", "uc_submit",  "assoc",   ""),
    ("a2", "actor_req", "uc_cancel",  "assoc",   ""),
    ("a3", "actor_adm", "uc_review",  "assoc",   ""),
    ("a4", "actor_adm", "uc_record",  "assoc",   ""),
    ("a5", "actor_adm", "uc_dash",    "assoc",   ""),
    ("i1", "uc_submit", "uc_avail",   "include", "«include»"),
    ("e1", "uc_risk",   "uc_review",  "extend",  "«extend»"),
    ("i2", "uc_risk",   "uc_explain", "include", "«include»"),
]
NOTE = ("Notes: «include» = mandatory sub-behaviour; «extend» = optional, advisory behaviour. "
        "Risk predictions never approve or reject a reservation automatically — the administrator decides (FR-DM-01 to FR-DM-03).")


def build_use_case_diagram():
    """Return (matplotlib figure, draw.io XML string, PNG bytes)."""
    EXIST = "ellipse;whiteSpace=wrap;html=1;fillColor=#dae8fc;strokeColor=#6c8ebf;fontSize=12;"
    NEW = "ellipse;whiteSpace=wrap;html=1;fillColor=#fff2cc;strokeColor=#d6b656;strokeWidth=2;fontSize=12;"
    ACTOR = "shape=umlActor;verticalLabelPosition=bottom;verticalAlign=top;html=1;outlineConnect=0;fontSize=12;fontStyle=1;"
    ASSOC = "endArrow=none;html=1;rounded=0;strokeWidth=1.5;"
    DEP = "endArrow=open;endSize=12;dashed=1;html=1;rounded=0;strokeWidth=1.5;fontSize=11;fontStyle=2;labelBackgroundColor=#ffffff;"

    cells = []

    def vertex(id_, value, style, x_, y_, w, h):
        cells.append(f'<mxCell id="{id_}" value={quoteattr(value)} style="{style}" vertex="1" parent="1">'
                     f'<mxGeometry x="{x_}" y="{y_}" width="{w}" height="{h}" as="geometry"/></mxCell>')

    def edge(id_, src, tgt, style, value=""):
        cells.append(f'<mxCell id="{id_}" value={quoteattr(value)} style="{style}" edge="1" parent="1" source="{src}" target="{tgt}">'
                     f'<mxGeometry relative="1" as="geometry"/></mxCell>')

    b_label, bx, by, bw, bh = BOUNDARY
    vertex("boundary", b_label, "rounded=0;whiteSpace=wrap;html=1;fillColor=none;strokeColor=#333333;strokeWidth=2;"
           "verticalAlign=top;fontStyle=1;fontSize=15;align=center;", bx, by, bw, bh)
    for k, (label, x_, y_) in ACTORS.items():
        vertex(k, label.replace("\n", "<br>"), ACTOR, x_, y_, 40, 80)
    for k, (label, x_, y_, is_new) in USE_CASES.items():
        vertex(k, label, NEW if is_new else EXIST, x_, y_, UC_W, UC_H)
    for id_, s, t, kind, label in LINKS:
        edge(id_, s, t, ASSOC if kind == "assoc" else DEP, label)
    vertex("leg_title", "Legend", "text;html=1;align=left;verticalAlign=middle;fontStyle=1;fontSize=12;", 215, 480, 80, 20)
    vertex("leg_old", "Existing use case", EXIST.replace("fontSize=12", "fontSize=10"), 215, 505, 130, 40)
    vertex("leg_new", "New analytical use case", NEW.replace("fontSize=12", "fontSize=10"), 365, 505, 150, 40)
    vertex("note", NOTE, "text;html=1;whiteSpace=wrap;align=left;verticalAlign=top;fontSize=11;fontStyle=2;", 540, 470, 305, 95)

    drawio_xml = ('<mxfile host="app.diagrams.net" agent="Streamlit" version="24.7.0" type="device">'
                  '<diagram id="uc-facility" name="Use Case Diagram">'
                  '<mxGraphModel dx="1250" dy="700" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" '
                  'fold="1" page="1" pageScale="1" pageWidth="1100" pageHeight="700" math="0" shadow="0"><root>'
                  '<mxCell id="0"/><mxCell id="1" parent="0"/>' + "".join(cells) + '</root></mxGraphModel></diagram></mxfile>')

    fig, ax = plt.subplots(figsize=(13.5, 8.2))
    ax.add_patch(mpatches.Rectangle((bx, by), bw, bh, fill=False, lw=2, ec="#333333"))
    ax.text(bx + bw / 2, by + 14, b_label, ha="center", va="top", fontsize=13, fontweight="bold")

    centers, radii = {}, {}
    for k, (label, x_, y_, is_new) in USE_CASES.items():
        cx, cy = x_ + UC_W / 2, y_ + UC_H / 2
        centers[k], radii[k] = (cx, cy), (UC_W / 2, UC_H / 2)
        ax.add_patch(mpatches.Ellipse((cx, cy), UC_W, UC_H, fc="#fff2cc" if is_new else "#dae8fc",
                                      ec="#d6b656" if is_new else "#6c8ebf", lw=2 if is_new else 1.2))
        ax.text(cx, cy, label, ha="center", va="center", fontsize=9, wrap=True)

    for k, (label, x_, y_) in ACTORS.items():
        cx = x_ + 20
        centers[k], radii[k] = (cx, y_ + 40), (14, 30)
        ax.add_patch(mpatches.Circle((cx, y_ + 10), 10, fill=False, lw=1.5))
        ax.plot([cx, cx], [y_ + 20, y_ + 50], "k", lw=1.5)
        ax.plot([cx - 18, cx + 18], [y_ + 30, y_ + 30], "k", lw=1.5)
        ax.plot([cx, cx - 16], [y_ + 50, y_ + 75], "k", lw=1.5)
        ax.plot([cx, cx + 16], [y_ + 50, y_ + 75], "k", lw=1.5)
        ax.text(cx, y_ + 88, label, ha="center", va="top", fontsize=9, fontweight="bold")

    def edge_point(a, b):
        (ax_, ay), (bx_, by_) = centers[a], centers[b]
        ra, rb = radii[a]
        dx, dy = bx_ - ax_, by_ - ay
        t = 1 / np.sqrt((dx / ra) ** 2 + (dy / rb) ** 2)
        return ax_ + dx * t, ay + dy * t

    for id_, s, t, kind, label in LINKS:
        p, q = edge_point(s, t), edge_point(t, s)
        if kind == "assoc":
            ax.plot([p[0], q[0]], [p[1], q[1]], "k", lw=1.4)
        else:
            ax.add_patch(mpatches.FancyArrowPatch(p, q, arrowstyle="->", mutation_scale=16, ls="--", lw=1.4, color="k"))
            ax.text((p[0] + q[0]) / 2, (p[1] + q[1]) / 2, label, ha="center", va="center", fontsize=9, style="italic",
                    bbox=dict(fc="white", ec="none", pad=1.5))

    ax.add_patch(mpatches.Ellipse((280, 525), 130, 40, fc="#dae8fc", ec="#6c8ebf"))
    ax.text(280, 525, "Existing use case", ha="center", va="center", fontsize=8)
    ax.add_patch(mpatches.Ellipse((440, 525), 150, 40, fc="#fff2cc", ec="#d6b656", lw=2))
    ax.text(440, 525, "New analytical use case", ha="center", va="center", fontsize=8)
    ax.text(215, 490, "Legend", fontsize=10, fontweight="bold", va="center")
    ax.text(540, 478, "\n".join(textwrap.wrap(NOTE, 62)), fontsize=8, style="italic", va="top")

    ax.set_xlim(20, 1010); ax.set_ylim(650, 10); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("UML Use Case Diagram — Campus Facility Reservation and Monitoring System (updated with analytical functions)",
                 fontsize=11, pad=8)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=170, bbox_inches="tight")
    return fig, drawio_xml, buf.getvalue()
