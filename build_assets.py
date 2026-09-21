#!/usr/bin/env python3
"""
Genera los SVG animados (SMIL, sin JavaScript) para el README de GitHub.

Uso:
    pip install matplotlib svgpath2mpl numpy scipy pillow
    python build_assets.py

Edita las secciones CONFIG de abajo para cambiar tecnologías, habilidades,
proyectos, educación o colores. Las rutas de los logos vienen de icons.json
(paths de simple-icons.org).
"""
import json
import math
import os
import random
from pathlib import Path

import numpy as np
from scipy.cluster.vq import kmeans2
from scipy.ndimage import binary_erosion
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).parent
OUT = Path(os.environ.get("OUT_DIR", HERE / "assets"))
OUT.mkdir(parents=True, exist_ok=True)

# ───────────────────────── CONFIG ─────────────────────────
# Paleta reducida a una única familia: cian (nada de dorado/rojos/morados/rosas).
# Solo variamos brillo/tono dentro del mismo hue para que cualquier transición
# de color (animate/interpolate) nunca cruce por verde o amarillo.
PALETTE = {
    "bg0": "#070b0d",
    "bg1": "#081418",
    "cyan": "#00f0ff",
    "cyan2": "#0a8f9c",
    "cyan3": "#8ff8ff",
    "dim": "#56728a",
    "ink": "#e6fbff",
}
NEON_CYCLE = ["cyan", "cyan3", "cyan2"]

MONO = "'JetBrains Mono','Fira Code','SFMono-Regular',Consolas,'Courier New',monospace"

# Radar (dominios) — valores 0-100. AJUSTA A TU NIVEL REAL.
RADAR = [
    ("ELECTRONICS", 85),
    ("EMBEDDED / IoT", 82),
    ("PYTHON", 92),
    ("DATA / BI", 85),
    ("AUTOMATION / RPA", 78),
    ("COMPUTER VISION", 72),
    ("CLOUD / DEVOPS", 55),
    ("SEMICONDUCTORS", 62),
]
# Barras (herramientas) — valores 0-100. AJUSTA A TU NIVEL REAL.
BARS = [
    ("Python", 92),
    ("SQL", 85),
    ("Excel (Advanced)", 88),
    ("PostgreSQL", 78),
    ("Django", 72),
    ("MATLAB", 68),
    ("OpenCV", 72),
    ("Docker", 58),
]

HEADER_LINES = [
    "Electronics Engineer",
    "Data Analyst & BI Specialist",
    "Embedded Systems & IoT",
    "Automation & Process Engineering",
]
NAME = "ABIMAEL FRANCO"

# About Me — texto corto, en líneas ya cortadas para el card.
ABOUT_LINES = [
    "Electronics Engineer with hands-on experience building technology",
    "solutions across automation, embedded systems and data analysis.",
    "Comfortable bridging hardware and software — sensors, microcontrollers",
    "and embedded Linux on one side; Python, SQL and BI dashboards on the other.",
    "Currently a Data & BI Strategy Specialist at BAC, automating reporting",
    "pipelines and building dashboards for operational decision-making.",
]
QUICK_FACTS = [
    ("ROLE", "Data & BI Strategy Specialist @ BAC"),
    ("STUDY", "B.Sc. Electronic Engineering, USAC"),
    ("ABROAD", "Semiconductor Training — Taiwan ICDF"),
    ("LANGUAGE", "English — B1+"),
]
ABOUT_TAGLINE = "turning technical requirements into working systems_"

# Educación formal y formación adicional.
FORMAL_EDU = [
    ("B.Sc. Electronic Engineering", "Universidad de San Carlos de Guatemala", "2020 – 2025", "Academic Excellence Award 2020 & 2021"),
    ("Postgraduate Specialization — Industrial Management", "Universidad de San Carlos de Guatemala", "2024 – 2025", ""),
    ("Computer Electronics Technician", "KINAL", "2017 – 2019", ""),
]
CERTS = [
    ("Semiconductor Technician Training", "Taiwan ICDF", "2025"),
    ("Sensors & Integrated Circuits Design/Fab.", "IEEE EDS / Universidad Galileo", "2024"),
    ("RPA Developer Foundation", "UiPath Academy", "2024"),
    ("AI & Machine Learning with Java", "INTECAP", "2022"),
    ("3D Studio MAX — Modeling Techniques", "INTECAP", "2019"),
    ("Cisco CCNA Routing & Switching", "KINAL", "2019"),
]

AWARDS = [
    ("Academic Excellence Award", "School of Engineering, USAC", "2020 & 2021"),
    ("Taiwan ICDF Scholarship", "IC Manufacturing & Packaging Program", "2025"),
    ("CONESIEE Congress Volunteer", "Engineering Academic Events", ""),
]

# Telemetría de GitHub — snapshot estático (datos reales, actualízalo re-corriendo
# el script). Sustituye a los widgets externos github-readme-stats / activity-graph,
# que dependen de una instancia pública de Vercel que se cae con frecuencia.
PROFILE_STATS = [
    ("PUBLIC REPOS", "47"),
    ("TOTAL STARS", "30"),
    ("FOLLOWERS", "6"),
    ("MEMBER SINCE", "2021"),
]
TOP_LANGS = [
    ("Python", 84.3),
    ("CSS", 4.3),
    ("HTML", 2.4),
    ("JavaScript", 2.3),
    ("Cython", 1.9),
]
# ──────────────────────────────────────────────────────────

C = PALETTE
random.seed(7)
np.random.seed(7)


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def num(v, d=1):
    s = f"{v:.{d}f}"
    return s.rstrip("0").rstrip(".") if "." in s else s


def check_times(times):
    assert times[0] == 0 and abs(times[-1] - 1) < 1e-9, times[:3]
    assert all(b >= a for a, b in zip(times, times[1:])), "keyTimes no monotónicos"


# ═══════════════════════════════════════════════════════════
#  0) CHROME COMÚN (fondo, grid, marco de esquinas cortadas)
# ═══════════════════════════════════════════════════════════
def card_defs(card_id, W, H, cut=26):
    return (
        f'<clipPath id="{card_id}"><polygon points="0,0 {W-cut},0 {W},{cut} {W},{H} {cut},{H} 0,{H-cut}"/></clipPath>'
    )


def card_frame(W, H, cut=26, col_a=None, col_b=None):
    col_a = col_a or C["cyan"]
    col_b = col_b or C["cyan3"]
    return (
        f'<polygon points="1,1 {W-cut-1},1 {W-1},{cut+1} {W-1},{H-1} {cut+1},{H-1} 1,{H-cut-1}" '
        f'fill="none" stroke="{col_a}" stroke-opacity=".5" stroke-width="1.2"/>'
        f'<path d="M1 {cut+34}V1h{cut+34}" fill="none" stroke="{col_b}" stroke-width="2.5"/>'
        f'<path d="M{W-1} {H-cut-34}V{H-1}H{W-cut-34}" fill="none" stroke="{col_a}" stroke-width="2.5"/>'
    )


def section_title_svg(kicker, tag="LOADED"):
    """Barra de título con el mismo fondo/marco que el resto de cards, para que
    los encabezados de sección nunca vuelvan al texto plano de GitHub."""
    W, H = 1000, 64
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-label="{esc(kicker)}">
<defs>
  <filter id="glowS" x="-20%" y="-50%" width="140%" height="200%"><feGaussianBlur stdDeviation="1.4" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
  <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="{C["bg0"]}"/><stop offset="1" stop-color="{C["bg1"]}"/></linearGradient>
  <pattern id="grid" width="28" height="28" patternUnits="userSpaceOnUse"><path d="M28 0H0V28" fill="none" stroke="{C["cyan"]}" stroke-opacity=".05"/></pattern>
  {card_defs("card", W, H, cut=16)}
</defs>
<g clip-path="url(#card)">
  <rect width="{W}" height="{H}" fill="url(#bg)"/>
  <rect width="{W}" height="{H}" fill="url(#grid)"/>
  <text x="36" y="{H//2+6}" font-family="{MONO}" font-size="16" letter-spacing="4" fill="{C["cyan"]}" filter="url(#glowS)">// {esc(kicker.upper())}</text>
  <g font-family="{MONO}" font-size="10.5" letter-spacing="2" fill="{C["cyan3"]}" text-anchor="end">
    <circle cx="{W-96}" cy="{H//2-3}" r="3"><animate attributeName="opacity" values="1;.2;1" dur="1.4s" repeatCount="indefinite"/></circle>
    <text x="{W-36}" y="{H//2}">{esc(tag)}</text>
  </g>
</g>
{card_frame(W, H, cut=16)}
</svg>
'''


# ═══════════════════════════════════════════════════════════
#  1) PARTÍCULAS QUE SE TRANSFORMAN EN LOGOS
# ═══════════════════════════════════════════════════════════
def logo_mask(path_d, res=260, invert=False):
    """Rasteriza un path SVG (viewBox 0 0 24 24) a una máscara booleana."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.path as mpath
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.patches import PathPatch
    from svg.path import Close, Line, parse_path as parse_svg_path

    def to_mpl_path(d, curve_samples=14):
        segments = parse_svg_path(d)
        verts, codes = [], []
        last_end = None
        subpath_start = None

        def close_subpath():
            if subpath_start is not None:
                verts.append(subpath_start)
                codes.append(mpath.Path.CLOSEPOLY)

        for seg in segments:
            if last_end is None or abs(seg.start - last_end) > 1e-6:
                close_subpath()
                p0 = (seg.start.real, seg.start.imag)
                verts.append(p0)
                codes.append(mpath.Path.MOVETO)
                subpath_start = p0
            if isinstance(seg, (Line, Close)):
                verts.append((seg.end.real, seg.end.imag))
                codes.append(mpath.Path.LINETO)
            else:  # CubicBezier, QuadraticBezier, Arc -> flatten
                for t in np.linspace(0, 1, curve_samples)[1:]:
                    p = seg.point(t)
                    verts.append((p.real, p.imag))
                    codes.append(mpath.Path.LINETO)
            last_end = seg.end
        close_subpath()
        return mpath.Path(verts, codes)

    mpl_path = to_mpl_path(path_d)
    fig = Figure(figsize=(1, 1), dpi=res)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 24)
    ax.set_ylim(24, 0)  # y crece hacia abajo, como en SVG
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.add_patch(PathPatch(mpl_path, facecolor="black", edgecolor="none", linewidth=0))
    canvas.draw()
    buf = np.asarray(canvas.buffer_rgba())
    gray = buf[..., :3].mean(axis=2)
    m = gray < 128
    if invert:  # logos "recortados" (p. ej. GitHub): usar la silueta en vez del disco
        yy, xx = np.mgrid[0:res, 0:res]
        k = res / 24
        disc = (xx - res / 2) ** 2 + (yy - 12.3 * k) ** 2 < (11.4 * k) ** 2
        m = disc & ~m
    return m


def sample_points(mask, n, edge_ratio=0.48):
    """Puntos uniformes: parte en el contorno (define la forma) y parte en el relleno."""
    edge = mask & ~binary_erosion(mask, iterations=2)
    inner = binary_erosion(mask, iterations=2)
    n_edge = int(n * edge_ratio)
    n_in = n - n_edge

    def km(pixels, k):
        ys, xs = np.nonzero(pixels)
        pts = np.column_stack([xs, ys]).astype(float)
        if len(pts) > 6000:
            pts = pts[np.random.choice(len(pts), 6000, replace=False)]
        k = min(k, len(pts))
        cent, _ = kmeans2(pts, k, minit="++", seed=3)
        return cent

    pe = km(edge, n_edge)
    pi = km(inner if inner.sum() > n_in else mask, n_in)
    pts = np.vstack([pe, pi])
    while len(pts) < n:
        pts = np.vstack([pts, pts[np.random.randint(len(pts))] + np.random.randn(2)])
    return pts[:n]


def build_particles(icons, n=400, size=200, cx=300, cy=170, S=3.0, hold_base=1.5):
    N = len(icons)
    T = S * N
    sets = []
    for ic in icons:
        m = logo_mask(ic["path"], invert=ic.get("invert", False))
        pts = sample_points(m, n)
        pts = (pts / m.shape[0] - 0.5) * size + np.array([cx, cy])
        sets.append(pts)

    order = [np.arange(n)]
    aligned = [sets[0]]
    for k in range(1, N):
        prev = aligned[-1]
        d = np.linalg.norm(prev[:, None, :] - sets[k][None, :, :], axis=2)
        r, c = linear_sum_assignment(d)
        aligned.append(sets[k][c])
    pos = np.stack(aligned)  # (N, n, 2)

    colors = [C[NEON_CYCLE[k % len(NEON_CYCLE)]] for k in range(N)]

    ct, cv = [], []
    for k in range(N):
        ct += [k * S / T, (k * S + hold_base + 0.15) / T]
        cv += [colors[k], colors[k]]
    ct.append(1.0)
    cv.append(colors[0])
    check_times(ct)

    circles = []
    for i in range(n):
        times, xs, ys, splines = [0.0], [pos[0, i, 0]], [pos[0, i, 1]], []
        for k in range(N):
            a = pos[k, i]
            b = pos[(k + 1) % N, i]
            s = k * S + hold_base + random.random() * 0.30
            mv = 0.8 + random.random() * 0.35
            e = min(s + mv, (k + 1) * S - 0.02)
            mid_t = (s + e) / 2
            mid = (a + b) / 2 + np.random.randn(2) * 26
            splines.append("0 0 1 1")
            times.append(s / T); xs.append(a[0]); ys.append(a[1])
            splines.append("0.5 0 1 1")
            times.append(mid_t / T); xs.append(mid[0]); ys.append(mid[1])
            splines.append("0 0 0.5 1")
            times.append(e / T); xs.append(b[0]); ys.append(b[1])
        if times[-1] < 1.0:
            splines.append("0 0 1 1")
            times.append(1.0); xs.append(pos[0, i, 0]); ys.append(pos[0, i, 1])
        for j in range(1, len(times)):
            times[j] = max(times[j], times[j - 1])
        check_times(times)
        assert len(times) == len(xs) == len(ys) == len(splines) + 1

        kt = ";".join(num(t, 4) for t in times)
        ks = ";".join(splines)
        r = random.choice([1.2, 1.5, 1.8, 2.1])
        op = num(0.65 + random.random() * 0.35, 2)
        vals = ";".join(f"{num(x)} {num(y)}" for x, y in zip(xs, ys))
        circles.append(
            f'<circle r="{r}" opacity="{op}" transform="translate({num(xs[0])} {num(ys[0])})">'
            f'<animateTransform attributeName="transform" type="translate" dur="{num(T)}s" repeatCount="indefinite" '
            f'calcMode="spline" keyTimes="{kt}" keySplines="{ks}" values="{vals}"/></circle>'
        )
    return dict(n=n, N=N, T=T, S=S, colors=colors, circles=circles, ct=ct, cv=cv, pos=pos, cx=cx, cy=cy)


def window(k, S, T, N, eps=0.0004):
    a, b = k * S / T, (k + 1) * S / T
    kt, vals = [0.0], []
    if k == 0:
        kt += [b, b + eps, 1.0]
        vals = ["1", "1", "0", "0"]
    elif k == N - 1:
        kt += [a, a + eps, 1.0]
        vals = ["0", "0", "1", "1"]
    else:
        kt += [a, a + eps, b, b + eps, 1.0]
        vals = ["0", "0", "1", "1", "0", "0"]
    assert len(kt) == len(vals), (k, len(kt), len(vals))
    check_times(kt)
    return ";".join(num(t, 4) for t in kt), ";".join(vals)


def particles_svg(icons):
    W, H = 1000, 350
    P = build_particles(icons)
    N, T, S = P["N"], P["T"], P["S"]
    cx, cy = P["cx"], P["cy"]
    dur = f"{num(T)}s"

    rows = []
    row0, row_h = 74, 21.5
    for k, ic in enumerate(icons):
        y = row0 + k * row_h
        col = P["colors"][k]
        kt, vals = window(k, S, T, N)
        a, b = k * S / T, (k + 1) * S / T
        if k == 0:
            bkt = [0.0, b, b + 0.0004, 1.0]; bv = ["0", "96", "0", "0"]
        elif k == N - 1:
            bkt = [0.0, a, 1.0]; bv = ["0", "0", "96"]
        else:
            bkt = [0.0, a, b, b + 0.0004, 1.0]; bv = ["0", "0", "96", "0", "0"]
        check_times(bkt)
        rows.append(
            f'<g font-family="{MONO}" font-size="13">'
            f'<text x="660" y="{num(y)}" fill="{C["dim"]}" opacity=".75">{esc(ic["name"].upper())}</text>'
            f'<rect x="790" y="{num(y-8)}" width="96" height="3" fill="{C["dim"]}" opacity=".18"/>'
            f'<g fill="{col}">'
            f'<text x="646" y="{num(y)}" font-weight="700" opacity="0">&gt;'
            f'<animate attributeName="opacity" dur="{dur}" repeatCount="indefinite" keyTimes="{kt}" values="{vals}"/></text>'
            f'<text x="660" y="{num(y)}" font-weight="700" opacity="0" filter="url(#glowS)">{esc(ic["name"].upper())}'
            f'<animate attributeName="opacity" dur="{dur}" repeatCount="indefinite" keyTimes="{kt}" values="{vals}"/></text>'
            f'<rect x="790" y="{num(y-8)}" width="0" height="3">'
            f'<animate attributeName="width" dur="{dur}" repeatCount="indefinite" keyTimes="{";".join(num(t,4) for t in bkt)}" values="{";".join(bv)}"/></rect>'
            f"</g></g>"
        )

    labels = []
    for k, ic in enumerate(icons):
        kt, vals = window(k, S, T, N)
        col = P["colors"][k]
        labels.append(
            f'<g opacity="0" text-anchor="middle" font-family="{MONO}">'
            f'<animate attributeName="opacity" dur="{dur}" repeatCount="indefinite" keyTimes="{kt}" values="{vals}"/>'
            f'<text x="{cx}" y="{cy+126}" font-size="22" font-weight="800" letter-spacing="6" fill="{col}" filter="url(#glowS)">{esc(ic["name"].upper())}</text>'
            f'<text x="{cx}" y="{cy+148}" font-size="11" letter-spacing="3" fill="{C["dim"]}">MODULE {k+1:02d}/{N:02d} · LOADED</text>'
            f"</g>"
        )

    ck = ";".join(num(t, 4) for t in P["ct"])
    cv = ";".join(P["cv"])

    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-label="Particle animation morphing into the logos of my main technologies: {", ".join(i["name"] for i in icons)}">
<title>Tech stack — particle scan</title>
<defs>
  <filter id="glow" x="-30%" y="-30%" width="160%" height="160%"><feGaussianBlur stdDeviation="2.2" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
  <filter id="glowS" x="-20%" y="-50%" width="140%" height="200%"><feGaussianBlur stdDeviation="1.6" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
  <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="{C["bg0"]}"/><stop offset="1" stop-color="{C["bg1"]}"/></linearGradient>
  <linearGradient id="beam" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="{C["cyan"]}" stop-opacity="0"/><stop offset=".5" stop-color="{C["cyan"]}" stop-opacity=".16"/><stop offset="1" stop-color="{C["cyan"]}" stop-opacity="0"/></linearGradient>
  <pattern id="grid" width="28" height="28" patternUnits="userSpaceOnUse"><path d="M28 0H0V28" fill="none" stroke="{C["cyan"]}" stroke-opacity=".055"/></pattern>
  <pattern id="scan" width="4" height="4" patternUnits="userSpaceOnUse"><rect width="4" height="1" fill="#000" opacity=".28"/></pattern>
  {card_defs("card", W, H)}
</defs>

<g clip-path="url(#card)">
  <rect width="{W}" height="{H}" fill="url(#bg)"/>
  <rect width="{W}" height="{H}" fill="url(#grid)"/>
  <circle cx="{cx}" cy="{cy}" r="150" fill="{C["cyan"]}" opacity=".035"/>

  <g fill="none" stroke-width="1.2">
    <circle cx="{cx}" cy="{cy}" r="122" stroke="{C["cyan"]}" stroke-opacity=".45" stroke-dasharray="46 22 6 22">
      <animateTransform attributeName="transform" type="rotate" from="0 {cx} {cy}" to="360 {cx} {cy}" dur="28s" repeatCount="indefinite"/></circle>
    <circle cx="{cx}" cy="{cy}" r="134" stroke="{C["cyan3"]}" stroke-opacity=".4" stroke-dasharray="4 14 60 14">
      <animateTransform attributeName="transform" type="rotate" from="360 {cx} {cy}" to="0 {cx} {cy}" dur="40s" repeatCount="indefinite"/></circle>
    <circle cx="{cx}" cy="{cy}" r="146" stroke="{C["dim"]}" stroke-opacity=".35" stroke-dasharray="1 7"/>
  </g>

  <g fill="{C["cyan"]}" filter="url(#glow)">
    <animate attributeName="fill" dur="{dur}" repeatCount="indefinite" keyTimes="{ck}" values="{cv}"/>
    {"".join(P["circles"])}
  </g>

  {"".join(labels)}

  <line x1="600" y1="34" x2="600" y2="{H-34}" stroke="{C["cyan"]}" stroke-opacity=".25"/>
  <text x="646" y="50" font-family="{MONO}" font-size="12" fill="{C["dim"]}" letter-spacing="2">$ ls ./stack --active</text>
  {"".join(rows)}

  <text x="36" y="42" font-family="{MONO}" font-size="12" letter-spacing="3" fill="{C["cyan"]}" opacity=".85">// STACK.SCAN</text>
  <g font-family="{MONO}" font-size="11" letter-spacing="2" fill="{C["cyan3"]}" text-anchor="end">
    <circle cx="{W-150}" cy="38" r="3.5"><animate attributeName="opacity" values="1;.15;1" dur="1.4s" repeatCount="indefinite"/></circle>
    <text x="{W-40}" y="42">SYS.ONLINE</text>
  </g>
  <text x="36" y="{H-22}" font-family="{MONO}" font-size="10" letter-spacing="2" fill="{C["dim"]}">PARTICLES {P["n"]} · MORPH LOOP {num(T)}s</text>

  <rect width="{W}" height="{H}" fill="url(#scan)"/>
  <rect x="0" y="-60" width="{W}" height="60" fill="url(#beam)">
    <animate attributeName="y" values="-60;{H}" dur="7s" repeatCount="indefinite"/></rect>
</g>

{card_frame(W, H)}
</svg>
'''


# ═══════════════════════════════════════════════════════════
#  2) HEADER CON GLITCH + TYPING
# ═══════════════════════════════════════════════════════════
def header_svg():
    W, H = 1000, 300
    T = 3.4 * len(HEADER_LINES)
    fs = 24
    cw = fs * 0.6
    x0 = 92
    y_line = 232

    clips, texts, cursor_kt, cursor_x = [], [], [], []
    for i, line in enumerate(HEADER_LINES):
        wtxt = len(line) * cw
        a = i * 3.4
        kt = [0.0, a / T, (a + 1.5) / T, (a + 2.9) / T, (a + 3.4) / T, 1.0]
        for j in range(1, len(kt)):
            kt[j] = max(kt[j], kt[j - 1])
        check_times(kt)
        vals = ["0", "0", num(wtxt), num(wtxt), "0", "0"]
        clips.append(
            f'<clipPath id="tc{i}"><rect x="{x0}" y="{y_line-fs}" height="{fs+10}" width="0">'
            f'<animate attributeName="width" dur="{num(T)}s" repeatCount="indefinite" keyTimes="{";".join(num(t,4) for t in kt)}" values="{";".join(vals)}"/></rect></clipPath>'
        )
        texts.append(
            f'<text x="{x0}" y="{y_line}" font-size="{fs}" fill="{C["cyan"]}" textLength="{num(wtxt)}" lengthAdjust="spacing" clip-path="url(#tc{i})">{esc(line)}</text>'
        )
        for t, xv in ((a, x0), (a + 1.5, x0 + wtxt), (a + 2.9, x0 + wtxt), (a + 3.4, x0)):
            cursor_kt.append(min(t / T, 1.0)); cursor_x.append(xv)
    cursor_kt[-1] = 1.0
    for j in range(1, len(cursor_kt)):
        cursor_kt[j] = max(cursor_kt[j], cursor_kt[j - 1])
    assert len(cursor_kt) == len(cursor_x)
    check_times(cursor_kt)

    gk = [0, .52, .521, .535, .536, .55, .551, .80, .801, .812, .813, 1.0]
    gx_m = [0, 0, 7, -5, 4, 0, 0, 0, -6, 5, 0, 0]
    gx_c = [0, 0, -7, 5, -4, 0, 0, 0, 6, -5, 0, 0]
    go = [0, 0, .9, .9, .9, 0, 0, 0, .9, .9, 0, 0]
    check_times(gk)
    gkt = ";".join(num(t, 3) for t in gk)
    nx = 60

    def glitch_layer(color, dx):
        xs = ";".join(num(nx + d) for d in dx)
        ops = ";".join(num(o, 2) for o in go)
        return (
            f'<text x="{nx}" y="150" font-size="76" font-weight="800" fill="{color}" opacity="0" textLength="800" lengthAdjust="spacingAndGlyphs">{NAME}'
            f'<animate attributeName="x" dur="7s" repeatCount="indefinite" calcMode="discrete" keyTimes="{gkt}" values="{xs}"/>'
            f'<animate attributeName="opacity" dur="7s" repeatCount="indefinite" calcMode="discrete" keyTimes="{gkt}" values="{ops}"/></text>'
        )

    traces = [
        ("M700 60H790L820 90H960", C["cyan"], 0),
        ("M760 250H850L880 220H960V170", C["cyan3"], 1.3),
        ("M660 30H730L750 50H940", C["cyan2"], 2.4),
    ]
    tr = ""
    for d, col, delay in traces:
        tr += (
            f'<path d="{d}" fill="none" stroke="{col}" stroke-opacity=".22" stroke-width="1.2"/>'
            f'<path d="{d}" fill="none" stroke="{col}" stroke-width="2" stroke-linecap="round" stroke-dasharray="26 300" filter="url(#glowS)">'
            f'<animate attributeName="stroke-dashoffset" from="326" to="0" dur="4.2s" begin="{delay}s" repeatCount="indefinite"/></path>'
        )
    nodes = "".join(
        f'<circle cx="{x}" cy="{y}" r="3" fill="{c}" opacity=".7"/>'
        for x, y, c in [(960, 90, C["cyan"]), (960, 170, C["cyan3"]), (940, 50, C["cyan2"])]
    )

    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-label="Abimael Franco — Electronics Engineer, Data Analyst &amp; BI Specialist">
<title>Abimael Franco</title>
<defs>
  <filter id="glow" x="-10%" y="-40%" width="120%" height="180%"><feGaussianBlur stdDeviation="4" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
  <filter id="glowS" x="-20%" y="-50%" width="140%" height="200%"><feGaussianBlur stdDeviation="2" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
  <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="{C["bg0"]}"/><stop offset=".6" stop-color="{C["bg1"]}"/><stop offset="1" stop-color="#141007"/></linearGradient>
  <linearGradient id="rule" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="{C["cyan3"]}"/><stop offset=".5" stop-color="{C["cyan"]}"/><stop offset="1" stop-color="{C["cyan2"]}"/></linearGradient>
  <pattern id="grid" width="32" height="32" patternUnits="userSpaceOnUse"><path d="M32 0H0V32" fill="none" stroke="{C["cyan"]}" stroke-opacity=".05"/></pattern>
  <pattern id="scan" width="4" height="4" patternUnits="userSpaceOnUse"><rect width="4" height="1" fill="#000" opacity=".28"/></pattern>
  <linearGradient id="beam" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="{C["cyan3"]}" stop-opacity="0"/><stop offset=".5" stop-color="{C["cyan3"]}" stop-opacity=".14"/><stop offset="1" stop-color="{C["cyan3"]}" stop-opacity="0"/></linearGradient>
  {card_defs("card", W, H, cut=30)}
  {"".join(clips)}
</defs>

<g clip-path="url(#card)">
  <rect width="{W}" height="{H}" fill="url(#bg)"/>
  <rect width="{W}" height="{H}" fill="url(#grid)"/>
  {tr}{nodes}

  <text x="{nx}" y="52" font-family="{MONO}" font-size="12" letter-spacing="3" fill="{C["cyan3"]}" opacity=".9">&gt; whoami<tspan fill="{C["dim"]}"> --profile --verbose</tspan></text>

  <g font-family="'Arial Black','Impact','Segoe UI',Arial,sans-serif">
    {glitch_layer(C["cyan3"], gx_m)}
    {glitch_layer(C["cyan"], gx_c)}
    <text x="{nx}" y="150" font-size="76" font-weight="800" fill="{C["ink"]}" textLength="800" lengthAdjust="spacingAndGlyphs" filter="url(#glow)">{NAME}</text>
  </g>

  <rect x="{nx}" y="172" width="800" height="3" fill="url(#rule)" filter="url(#glowS)"/>

  <g font-family="{MONO}">
    <text x="{nx}" y="{y_line}" font-size="{fs}" fill="{C["cyan3"]}" font-weight="700">&gt;</text>
    <g filter="url(#glowS)">{"".join(texts)}</g>
    <rect x="{x0}" y="{y_line-fs+2}" width="12" height="{fs}" fill="{C["ink"]}">
      <animate attributeName="x" dur="{num(T)}s" repeatCount="indefinite" keyTimes="{";".join(num(t,4) for t in cursor_kt)}" values="{";".join(num(v) for v in cursor_x)}"/>
      <animate attributeName="opacity" values="1;1;0;0" keyTimes="0;.5;.5001;1" dur=".9s" repeatCount="indefinite"/>
    </rect>
  </g>

  <text x="{nx}" y="{H-26}" font-family="{MONO}" font-size="11" letter-spacing="3" fill="{C["dim"]}">HARDWARE // SOFTWARE // EMBEDDED // DATA // AI</text>

  <rect width="{W}" height="{H}" fill="url(#scan)"/>
  <rect x="0" y="-70" width="{W}" height="70" fill="url(#beam)"><animate attributeName="y" values="-70;{H}" dur="6s" repeatCount="indefinite"/></rect>
</g>

{card_frame(W, H, cut=30)}
</svg>
'''


# ═══════════════════════════════════════════════════════════
#  3) SKILLS: RADAR + BARRAS
# ═══════════════════════════════════════════════════════════
def skills_svg():
    W, H = 1000, 400
    T = 10.0
    cx, cy, R = 268, 212, 110
    n = len(RADAR)

    def pt(i, v):
        ang = -math.pi / 2 + i * 2 * math.pi / n
        r = R * v / 100
        return cx + r * math.cos(ang), cy + r * math.sin(ang)

    rings = ""
    for lvl in (25, 50, 75, 100):
        pts = " ".join(f"{num(x)},{num(y)}" for x, y in (pt(i, lvl) for i in range(n)))
        rings += f'<polygon points="{pts}" fill="none" stroke="{C["cyan"]}" stroke-opacity="{.32 if lvl==100 else .14}"/>'
    axes = "".join(
        f'<line x1="{cx}" y1="{cy}" x2="{num(pt(i,100)[0])}" y2="{num(pt(i,100)[1])}" stroke="{C["cyan"]}" stroke-opacity=".14"/>'
        for i in range(n)
    )
    labels = ""
    for i, (name, v) in enumerate(RADAR):
        x, y = pt(i, 100)
        ang = -math.pi / 2 + i * 2 * math.pi / n
        lx, ly = cx + (R + 16) * math.cos(ang), cy + (R + 16) * math.sin(ang)
        anchor = "middle" if abs(math.cos(ang)) < 0.25 else ("start" if math.cos(ang) > 0 else "end")
        ly += 4 if math.sin(ang) > -0.9 else 0
        labels += f'<text x="{num(lx)}" y="{num(ly)}" text-anchor="{anchor}" font-size="11" letter-spacing="1.5" fill="{C["ink"]}" opacity=".85">{esc(name)}</text>'

    center_pts = " ".join(f"{cx},{cy}" for _ in range(n))
    full_pts = " ".join(f"{num(x)},{num(y)}" for x, y in (pt(i, v) for i, (_, v) in enumerate(RADAR)))
    kt = [0, .12, .88, 1.0]
    kts = ";".join(num(t, 3) for t in kt)
    spl = "0.2 0.8 0.2 1;0 0 1 1;0.6 0 0.8 0.2"
    poly = (
        f'<polygon points="{center_pts}" fill="{C["cyan3"]}" fill-opacity=".2" stroke="{C["cyan3"]}" stroke-width="2" stroke-linejoin="round" filter="url(#glowS)">'
        f'<animate attributeName="points" dur="{T}s" repeatCount="indefinite" calcMode="spline" keyTimes="{kts}" keySplines="{spl}" values="{center_pts};{full_pts};{full_pts};{center_pts}"/></polygon>'
    )
    dots = ""
    for i, (_, v) in enumerate(RADAR):
        x, y = pt(i, v)
        dots += (
            f'<circle cx="{cx}" cy="{cy}" r="3.4" fill="{C["cyan"]}" filter="url(#glowS)">'
            f'<animate attributeName="cx" dur="{T}s" repeatCount="indefinite" calcMode="spline" keyTimes="{kts}" keySplines="{spl}" values="{cx};{num(x)};{num(x)};{cx}"/>'
            f'<animate attributeName="cy" dur="{T}s" repeatCount="indefinite" calcMode="spline" keyTimes="{kts}" keySplines="{spl}" values="{cy};{num(y)};{num(y)};{cy}"/></circle>'
        )

    bx, bw, row0, rh = 610, 270, 96, 36
    bars = ""
    for i, (name, v) in enumerate(BARS):
        y = row0 + i * rh
        ds = i * 0.16
        w = bw * v / 100
        bkt = [0.0, ds / T, (ds + 1.3) / T, 0.9, 1.0]
        check_times(bkt)
        bk = ";".join(num(t, 4) for t in bkt)
        bars += (
            f'<g font-family="{MONO}">'
            f'<text x="{bx-14}" y="{y+4}" text-anchor="end" font-size="13" fill="{C["ink"]}" opacity=".9">{esc(name)}</text>'
            f'<rect x="{bx}" y="{y-6}" width="{bw}" height="10" fill="{C["cyan"]}" fill-opacity=".07" stroke="{C["cyan"]}" stroke-opacity=".28"/>'
            f'<rect x="{bx}" y="{y-6}" width="0" height="10" fill="url(#bar)" filter="url(#glowS)">'
            f'<animate attributeName="width" dur="{T}s" repeatCount="indefinite" calcMode="spline" keyTimes="{bk}" keySplines="0 0 1 1;0.2 0.8 0.2 1;0 0 1 1;0 0 1 1" values="0;0;{num(w)};{num(w)};0"/></rect>'
            f'<text x="{bx+bw+12}" y="{y+4}" font-size="12" fill="{C["cyan3"]}" opacity="0">{v}%'
            f'<animate attributeName="opacity" dur="{T}s" repeatCount="indefinite" keyTimes="{bk}" values="0;0;1;1;0"/></text>'
            f"</g>"
        )

    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-label="Skill radar and tool proficiency bars">
<title>Skill matrix</title>
<defs>
  <filter id="glowS" x="-20%" y="-50%" width="140%" height="200%"><feGaussianBlur stdDeviation="2" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
  <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="{C["bg0"]}"/><stop offset="1" stop-color="{C["bg1"]}"/></linearGradient>
  <linearGradient id="bar" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="{C["cyan"]}"/><stop offset="1" stop-color="{C["cyan3"]}"/></linearGradient>
  <pattern id="grid" width="28" height="28" patternUnits="userSpaceOnUse"><path d="M28 0H0V28" fill="none" stroke="{C["cyan"]}" stroke-opacity=".05"/></pattern>
  <pattern id="scan" width="4" height="4" patternUnits="userSpaceOnUse"><rect width="4" height="1" fill="#000" opacity=".28"/></pattern>
  {card_defs("card", W, H)}
</defs>
<g clip-path="url(#card)" font-family="{MONO}">
  <rect width="{W}" height="{H}" fill="url(#bg)"/>
  <rect width="{W}" height="{H}" fill="url(#grid)"/>
  <text x="36" y="42" font-size="12" letter-spacing="3" fill="{C["cyan"]}" opacity=".85">// SKILL.MATRIX</text>
  <text x="520" y="42" font-size="12" letter-spacing="3" fill="{C["cyan3"]}" opacity=".9">// TOOLCHAIN.LOAD</text>
  <line x1="500" y1="34" x2="500" y2="{H-34}" stroke="{C["cyan"]}" stroke-opacity=".22"/>
  {rings}{axes}{poly}{dots}{labels}
  {bars}
  <rect width="{W}" height="{H}" fill="url(#scan)"/>
</g>
{card_frame(W, H)}
</svg>
'''


# ═══════════════════════════════════════════════════════════
#  4) DIVISOR
# ═══════════════════════════════════════════════════════════
def divider_svg():
    W, H = 1000, 24
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="separator" aria-hidden="true">
<defs>
  <linearGradient id="l" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="{C["cyan3"]}" stop-opacity="0"/><stop offset=".25" stop-color="{C["cyan3"]}"/><stop offset=".5" stop-color="{C["cyan"]}"/><stop offset=".75" stop-color="{C["cyan2"]}"/><stop offset="1" stop-color="{C["cyan2"]}" stop-opacity="0"/></linearGradient>
  <linearGradient id="g" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="#fff" stop-opacity="0"/><stop offset=".5" stop-color="#fff"/><stop offset="1" stop-color="#fff" stop-opacity="0"/></linearGradient>
  <filter id="glow" x="-10%" y="-300%" width="120%" height="700%"><feGaussianBlur stdDeviation="2.5" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
</defs>
<rect width="{W}" height="{H}" fill="{C["bg0"]}"/>
<rect x="60" y="11" width="{W-120}" height="2" fill="url(#l)" filter="url(#glow)"/>
<rect x="-160" y="10" width="160" height="4" fill="url(#g)" opacity=".9"><animate attributeName="x" values="-160;{W}" dur="3.6s" repeatCount="indefinite"/></rect>
<g transform="translate(500 12) rotate(45)"><rect x="-5" y="-5" width="10" height="10" fill="{C["bg0"]}" stroke="{C["cyan3"]}" stroke-width="1.5"/></g>
<circle cx="500" cy="12" r="1.8" fill="{C["cyan3"]}"><animate attributeName="opacity" values="1;.2;1" dur="1.6s" repeatCount="indefinite"/></circle>
</svg>
'''


# ═══════════════════════════════════════════════════════════
#  5) ABOUT ME — card estilo terminal
# ═══════════════════════════════════════════════════════════
def about_svg():
    W, H = 1000, 300

    lines = ""
    for i, line in enumerate(ABOUT_LINES):
        y = 96 + i * 24
        delay = i * 0.18
        lines += (
            f'<text x="60" y="{y}" font-family="{MONO}" font-size="14.5" fill="{C["ink"]}" opacity="0">{esc(line)}'
            f'<animate attributeName="opacity" values="0;1" dur=".6s" begin="{num(delay,2)}s" fill="freeze"/></text>'
        )

    facts = ""
    fx, fy0, frh = 668, 100, 44
    for i, (label, value) in enumerate(QUICK_FACTS):
        y = fy0 + i * frh
        delay = 0.4 + i * 0.15
        facts += (
            f'<g opacity="0"><animate attributeName="opacity" values="0;1" dur=".5s" begin="{num(delay,2)}s" fill="freeze"/>'
            f'<rect x="{fx}" y="{y-11}" width="7" height="7" fill="{C["cyan3"]}"/>'
            f'<text x="{fx+16}" y="{y-3}" font-family="{MONO}" font-size="10.5" letter-spacing="2" fill="{C["cyan"]}">{esc(label)}</text>'
            f'<text x="{fx+16}" y="{y+13}" font-family="{MONO}" font-size="12.5" fill="{C["ink"]}">{esc(value)}</text>'
            f"</g>"
        )

    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-label="About me: Electronics Engineer and Data &amp; BI Specialist summary">
<title>About me</title>
<defs>
  <filter id="glowS" x="-20%" y="-50%" width="140%" height="200%"><feGaussianBlur stdDeviation="1.6" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
  <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="{C["bg0"]}"/><stop offset="1" stop-color="{C["bg1"]}"/></linearGradient>
  <pattern id="grid" width="28" height="28" patternUnits="userSpaceOnUse"><path d="M28 0H0V28" fill="none" stroke="{C["cyan"]}" stroke-opacity=".05"/></pattern>
  <pattern id="scan" width="4" height="4" patternUnits="userSpaceOnUse"><rect width="4" height="1" fill="#000" opacity=".28"/></pattern>
  {card_defs("card", W, H)}
</defs>
<g clip-path="url(#card)">
  <rect width="{W}" height="{H}" fill="url(#bg)"/>
  <rect width="{W}" height="{H}" fill="url(#grid)"/>
  <text x="36" y="42" font-family="{MONO}" font-size="12" letter-spacing="3" fill="{C["cyan"]}" opacity=".85">// ABOUT.ME</text>
  <text x="36" y="66" font-family="{MONO}" font-size="12" fill="{C["cyan3"]}" opacity=".9">&gt; cat bio.md</text>
  <g filter="url(#glowS)">{lines}</g>

  <line x1="638" y1="34" x2="638" y2="{H-40}" stroke="{C["cyan"]}" stroke-opacity=".22"/>
  <text x="668" y="66" font-family="{MONO}" font-size="12" letter-spacing="3" fill="{C["cyan3"]}" opacity=".9">// QUICK.FACTS</text>
  {facts}

  <rect x="36" y="{H-46}" width="{W-72}" height="1" fill="{C["cyan"]}" opacity=".2"/>
  <text x="36" y="{H-22}" font-family="{MONO}" font-size="12" fill="{C["dim"]}">&gt; <tspan fill="{C["cyan3"]}">{esc(ABOUT_TAGLINE)}</tspan>
    <animate attributeName="opacity" values="1;1;0;0" keyTimes="0;.5;.5001;1" dur="1s" repeatCount="indefinite"/>
  </text>

  <rect width="{W}" height="{H}" fill="url(#scan)"/>
</g>
{card_frame(W, H)}
</svg>
'''


# ═══════════════════════════════════════════════════════════
#  6) EDUCATION — timeline + certificaciones
# ═══════════════════════════════════════════════════════════
def education_svg():
    W, H = 1000, 480

    tx, ty0, trh = 90, 118, 148
    line_len = ty0 + (len(FORMAL_EDU) - 1) * trh
    timeline = f'<line x1="{tx}" y1="{ty0}" x2="{tx}" y2="{line_len}" stroke="{C["cyan"]}" stroke-opacity=".35" stroke-width="2" stroke-dasharray="{line_len-ty0}">'
    timeline += f'<animate attributeName="stroke-dashoffset" from="{line_len-ty0}" to="0" dur="1.4s" fill="freeze"/></line>'

    nodes = ""
    for i, (title, org, year, note) in enumerate(FORMAL_EDU):
        y = ty0 + i * trh
        delay = 0.3 + i * 0.25
        block = (
            f'<g opacity="0"><animate attributeName="opacity" values="0;1" dur=".6s" begin="{num(delay,2)}s" fill="freeze"/>'
            f'<circle cx="{tx}" cy="{y}" r="7" fill="{C["bg0"]}" stroke="{C["cyan3"]}" stroke-width="2.5" filter="url(#glowS)"/>'
            f'<circle cx="{tx}" cy="{y}" r="2.3" fill="{C["cyan3"]}"/>'
            f'<text x="{tx+26}" y="{y-8}" font-family="{MONO}" font-size="14" font-weight="700" fill="{C["ink"]}">{esc(title)}</text>'
            f'<text x="{tx+26}" y="{y+11}" font-family="{MONO}" font-size="11.5" fill="{C["dim"]}">{esc(org)}</text>'
            f'<text x="{tx+26}" y="{y+28}" font-family="{MONO}" font-size="11" fill="{C["cyan"]}">{esc(year)}</text>'
        )
        if note:
            block += f'<text x="{tx+26}" y="{y+45}" font-family="{MONO}" font-size="10.5" fill="{C["cyan3"]}">★ {esc(note)}</text>'
        block += "</g>"
        nodes += block

    cx0, cy0, crh = 560, 112, 56
    certs = ""
    for i, (title, org, year) in enumerate(CERTS):
        y = cy0 + i * crh
        delay = 0.3 + i * 0.12
        certs += (
            f'<g opacity="0"><animate attributeName="opacity" values="0;1" dur=".5s" begin="{num(delay,2)}s" fill="freeze"/>'
            f'<rect x="{cx0}" y="{y-16}" width="8" height="8" fill="{C["cyan"]}"/>'
            f'<text x="{cx0+18}" y="{y-8}" font-family="{MONO}" font-size="12.5" font-weight="700" fill="{C["ink"]}">{esc(title)}</text>'
            f'<text x="{cx0+18}" y="{y+9}" font-family="{MONO}" font-size="11" fill="{C["dim"]}">{esc(org)}</text>'
            f'<rect x="{W-100}" y="{y-20}" width="60" height="20" fill="{C["cyan3"]}" opacity=".12" stroke="{C["cyan3"]}" stroke-opacity=".5"/>'
            f'<text x="{W-70}" y="{y-6}" text-anchor="middle" font-family="{MONO}" font-size="11" fill="{C["cyan3"]}">{esc(year)}</text>'
            f"</g>"
        )

    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-label="Education timeline and additional certifications">
<title>Education &amp; training</title>
<defs>
  <filter id="glowS" x="-20%" y="-50%" width="140%" height="200%"><feGaussianBlur stdDeviation="1.6" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
  <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="{C["bg0"]}"/><stop offset="1" stop-color="{C["bg1"]}"/></linearGradient>
  <pattern id="grid" width="28" height="28" patternUnits="userSpaceOnUse"><path d="M28 0H0V28" fill="none" stroke="{C["cyan"]}" stroke-opacity=".05"/></pattern>
  <pattern id="scan" width="4" height="4" patternUnits="userSpaceOnUse"><rect width="4" height="1" fill="#000" opacity=".28"/></pattern>
  {card_defs("card", W, H)}
</defs>
<g clip-path="url(#card)">
  <rect width="{W}" height="{H}" fill="url(#bg)"/>
  <rect width="{W}" height="{H}" fill="url(#grid)"/>
  <text x="36" y="42" font-family="{MONO}" font-size="12" letter-spacing="3" fill="{C["cyan"]}" opacity=".85">// EDUCATION</text>
  <text x="536" y="42" font-family="{MONO}" font-size="12" letter-spacing="3" fill="{C["cyan3"]}" opacity=".9">// CERTIFICATIONS.LOG</text>
  <line x1="500" y1="34" x2="500" y2="{H-30}" stroke="{C["cyan"]}" stroke-opacity=".22"/>
  {timeline}
  {nodes}
  {certs}
  <rect width="{W}" height="{H}" fill="url(#scan)"/>
</g>
{card_frame(W, H)}
</svg>
'''


# ═══════════════════════════════════════════════════════════
#  7) AWARDS — medallas
# ═══════════════════════════════════════════════════════════
def _medal(cx, cy, col):
    star = ""
    pts = []
    for k in range(10):
        ang = -math.pi / 2 + k * math.pi / 5
        r = 8 if k % 2 == 0 else 3.4
        pts.append(f"{num(cx+r*math.cos(ang))},{num(cy+r*math.sin(ang))}")
    star = f'<polygon points="{" ".join(pts)}" fill="{col}"/>'
    return (
        f'<polygon points="{cx-16},{cy+34} {cx-6},{cy+4} {cx-2},{cy+8} {cx-12},{cy+40}" fill="{C["cyan"]}" opacity=".85"/>'
        f'<polygon points="{cx+16},{cy+34} {cx+6},{cy+4} {cx+2},{cy+8} {cx+12},{cy+40}" fill="{C["cyan2"]}" opacity=".85"/>'
        f'<circle cx="{cx}" cy="{cy}" r="24" fill="{C["bg1"]}" stroke="{col}" stroke-width="2.5" filter="url(#glowS)">'
        f'<animate attributeName="r" values="24;26;24" dur="2.4s" repeatCount="indefinite"/></circle>'
        f'<circle cx="{cx}" cy="{cy}" r="17" fill="none" stroke="{col}" stroke-opacity=".5"/>'
        f"{star}"
    )


def awards_svg():
    W, H = 1000, 240
    centers = [180, 500, 820]
    cy = 78
    cards = ""
    for i, ((title, org, year), cx) in enumerate(zip(AWARDS, centers)):
        delay = i * 0.25
        col = C["cyan3"] if i % 2 == 0 else C["cyan"]
        label = f"{title}" if not year else f"{title}  ·  {year}"
        cards += (
            f'<g opacity="0"><animate attributeName="opacity" values="0;1" dur=".6s" begin="{num(delay,2)}s" fill="freeze"/>'
            f'{_medal(cx, cy, col)}'
            f'<text x="{cx}" y="{cy+64}" text-anchor="middle" font-family="{MONO}" font-size="13.5" font-weight="700" fill="{C["ink"]}">{esc(title)}</text>'
            f'<text x="{cx}" y="{cy+84}" text-anchor="middle" font-family="{MONO}" font-size="11" fill="{C["dim"]}">{esc(org)}</text>'
        )
        if year:
            cards += f'<text x="{cx}" y="{cy+102}" text-anchor="middle" font-family="{MONO}" font-size="11" fill="{col}">{esc(year)}</text>'
        cards += "</g>"

    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-label="Awards and recognitions">
<title>Awards &amp; recognition</title>
<defs>
  <filter id="glowS" x="-30%" y="-30%" width="160%" height="160%"><feGaussianBlur stdDeviation="1.8" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
  <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="{C["bg0"]}"/><stop offset="1" stop-color="{C["bg1"]}"/></linearGradient>
  <pattern id="grid" width="28" height="28" patternUnits="userSpaceOnUse"><path d="M28 0H0V28" fill="none" stroke="{C["cyan"]}" stroke-opacity=".05"/></pattern>
  <pattern id="scan" width="4" height="4" patternUnits="userSpaceOnUse"><rect width="4" height="1" fill="#000" opacity=".28"/></pattern>
  {card_defs("card", W, H)}
</defs>
<g clip-path="url(#card)">
  <rect width="{W}" height="{H}" fill="url(#bg)"/>
  <rect width="{W}" height="{H}" fill="url(#grid)"/>
  <text x="36" y="40" font-family="{MONO}" font-size="12" letter-spacing="3" fill="{C["cyan"]}" opacity=".85">// AWARDS.LOG</text>
  <line x1="340" y1="34" x2="340" y2="{H-34}" stroke="{C["cyan"]}" stroke-opacity=".18"/>
  <line x1="660" y1="34" x2="660" y2="{H-34}" stroke="{C["cyan"]}" stroke-opacity=".18"/>
  {cards}
  <rect width="{W}" height="{H}" fill="url(#scan)"/>
</g>
{card_frame(W, H)}
</svg>
'''


# ═══════════════════════════════════════════════════════════
#  7b) GITHUB TELEMETRY — stats + lenguajes (datos reales, self-hosted)
# ═══════════════════════════════════════════════════════════
def telemetry_svg():
    W, H = 1000, 300

    sx, sy0, srh = 60, 110, 78
    stats = ""
    for i, (label, value) in enumerate(PROFILE_STATS):
        col, row = i % 2, i // 2
        x = sx + col * 260
        y = sy0 + row * srh
        stats += (
            f'<text x="{x}" y="{y}" font-family="{MONO}" font-size="30" font-weight="700" fill="{C["ink"]}" filter="url(#glowS)">{esc(value)}</text>'
            f'<text x="{x}" y="{y+20}" font-family="{MONO}" font-size="10.5" letter-spacing="2" fill="{C["cyan"]}">{esc(label)}</text>'
        )

    bx, bw, brow0, brh = 560, 340, 100, 34
    bars = ""
    for i, (name, pct) in enumerate(TOP_LANGS):
        y = brow0 + i * brh
        w = bw * min(pct, 100) / 100
        delay = i * 0.15
        bars += (
            f'<g font-family="{MONO}">'
            f'<text x="{bx}" y="{y-8}" font-size="12" fill="{C["ink"]}" opacity=".9">{esc(name)}</text>'
            f'<text x="{bx+bw}" y="{y-8}" text-anchor="end" font-size="11" fill="{C["cyan3"]}">{num(pct,1)}%</text>'
            f'<rect x="{bx}" y="{y-2}" width="{bw}" height="8" fill="{C["cyan"]}" fill-opacity=".08" stroke="{C["cyan"]}" stroke-opacity=".25"/>'
            f'<rect x="{bx}" y="{y-2}" width="0" height="8" fill="url(#bar)" filter="url(#glowS)" opacity="0">'
            f'<animate attributeName="width" values="0;{num(w)}" dur=".8s" begin="{num(delay,2)}s" fill="freeze"/>'
            f'<animate attributeName="opacity" values="0;1" dur=".3s" begin="{num(delay,2)}s" fill="freeze"/></rect>'
            f"</g>"
        )

    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-label="GitHub telemetry: public repos, stars, followers and top languages">
<title>GitHub telemetry</title>
<defs>
  <filter id="glowS" x="-20%" y="-50%" width="140%" height="200%"><feGaussianBlur stdDeviation="1.6" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
  <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="{C["bg0"]}"/><stop offset="1" stop-color="{C["bg1"]}"/></linearGradient>
  <linearGradient id="bar" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="{C["cyan"]}"/><stop offset="1" stop-color="{C["cyan3"]}"/></linearGradient>
  <pattern id="grid" width="28" height="28" patternUnits="userSpaceOnUse"><path d="M28 0H0V28" fill="none" stroke="{C["cyan"]}" stroke-opacity=".05"/></pattern>
  <pattern id="scan" width="4" height="4" patternUnits="userSpaceOnUse"><rect width="4" height="1" fill="#000" opacity=".28"/></pattern>
  {card_defs("card", W, H)}
</defs>
<g clip-path="url(#card)">
  <rect width="{W}" height="{H}" fill="url(#bg)"/>
  <rect width="{W}" height="{H}" fill="url(#grid)"/>
  <text x="36" y="42" font-family="{MONO}" font-size="12" letter-spacing="3" fill="{C["cyan"]}" opacity=".85">// PROFILE.STATS</text>
  <text x="536" y="42" font-family="{MONO}" font-size="12" letter-spacing="3" fill="{C["cyan"]}" opacity=".85">// TOP.LANGUAGES</text>
  <line x1="500" y1="34" x2="500" y2="{H-34}" stroke="{C["cyan"]}" stroke-opacity=".22"/>
  {stats}
  {bars}
  <text x="36" y="{H-22}" font-family="{MONO}" font-size="10" fill="{C["dim"]}">SOURCE: GITHUB API · SNAPSHOT</text>
  <rect width="{W}" height="{H}" fill="url(#scan)"/>
</g>
{card_frame(W, H)}
</svg>
'''


# ═══════════════════════════════════════════════════════════
#  8) PROYECTOS — thumbnails animados
# ═══════════════════════════════════════════════════════════
def _project_chrome(W, H, title, tag):
    return f'''<defs>
  <filter id="glowS" x="-30%" y="-30%" width="160%" height="160%"><feGaussianBlur stdDeviation="1.6" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
  <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="{C["bg0"]}"/><stop offset="1" stop-color="{C["bg1"]}"/></linearGradient>
  <pattern id="grid" width="24" height="24" patternUnits="userSpaceOnUse"><path d="M24 0H0V24" fill="none" stroke="{C["cyan"]}" stroke-opacity=".06"/></pattern>
  {card_defs("card", W, H, cut=18)}
</defs>
<g clip-path="url(#card)">
  <rect width="{W}" height="{H}" fill="url(#bg)"/>
  <rect width="{W}" height="{H}" fill="url(#grid)"/>
  <text x="20" y="28" font-family="{MONO}" font-size="11" letter-spacing="2" fill="{C["cyan"]}" opacity=".85">// {esc(tag)}</text>
  <g font-family="{MONO}" font-size="10" letter-spacing="1.5" fill="{C["cyan3"]}" text-anchor="end">
    <circle cx="{W-52}" cy="24" r="3"><animate attributeName="opacity" values="1;.2;1" dur="1.3s" repeatCount="indefinite"/></circle>
    <text x="{W-20}" y="28">ACTIVE</text>
  </g>'''


def robotic_arm_svg():
    W, H = 480, 260
    bx, by = 130, 226
    svg = _project_chrome(W, H, "6-DOF ROBOTIC ARM", "ARM.SIM")
    svg += f'<line x1="30" y1="{by}" x2="{W-30}" y2="{by}" stroke="{C["cyan"]}" stroke-opacity=".3"/>'
    svg += f'<rect x="{bx-24}" y="{by-6}" width="48" height="10" fill="{C["bg1"]}" stroke="{C["cyan"]}" stroke-opacity=".6"/>'
    svg += f'<g transform="translate({bx} {by})">'
    svg += (
        f'<g><animateTransform attributeName="transform" type="rotate" values="-18;22;-18" '
        f'keyTimes="0;.5;1" calcMode="spline" keySplines="0.4 0 0.2 1;0.4 0 0.2 1" dur="4.6s" repeatCount="indefinite"/>'
        f'<line x1="0" y1="0" x2="0" y2="-92" stroke="{C["cyan"]}" stroke-width="10" stroke-linecap="round"/>'
        f'<circle cx="0" cy="0" r="7" fill="{C["cyan3"]}"/>'
        f'<g transform="translate(0 -92)">'
        f'<animateTransform attributeName="transform" type="rotate" values="24;-34;24" additive="sum" '
        f'keyTimes="0;.5;1" calcMode="spline" keySplines="0.4 0 0.2 1;0.4 0 0.2 1" dur="4.6s" repeatCount="indefinite"/>'
        f'<line x1="0" y1="0" x2="0" y2="-72" stroke="{C["cyan2"]}" stroke-width="8" stroke-linecap="round"/>'
        f'<circle cx="0" cy="0" r="6" fill="{C["cyan3"]}"/>'
        f'<g transform="translate(0 -72)">'
        f'<animateTransform attributeName="transform" type="rotate" values="-14;46;-14" additive="sum" '
        f'keyTimes="0;.5;1" calcMode="spline" keySplines="0.4 0 0.2 1;0.4 0 0.2 1" dur="4.6s" repeatCount="indefinite"/>'
        f'<line x1="0" y1="0" x2="0" y2="-46" stroke="{C["cyan3"]}" stroke-width="6" stroke-linecap="round"/>'
        f'<circle cx="0" cy="-46" r="7" fill="{C["cyan3"]}" filter="url(#glowS)">'
        f'<animate attributeName="r" values="7;9;7" dur="1.4s" repeatCount="indefinite"/></circle>'
        f"</g></g></g>"
    )
    svg += "</g>"
    svg += f'<text x="20" y="{H-16}" font-family="{MONO}" font-size="10.5" fill="{C["dim"]}">MATLAB · CoppeliaSim · Raspberry Pi · OpenCV</text>'
    svg += "</g>"
    svg += card_frame(W, H, cut=18)
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-label="Animated 6 degrees of freedom robotic arm">{svg}</svg>\n'


def humidity_svg():
    W, H = 480, 260
    cx0, cy0, cw, ch = 40, 46, 190, 170
    svg = _project_chrome(W, H, "ENVIRONMENTAL CONTROL", "CHAMBER.MON")
    svg += f'<rect x="{cx0}" y="{cy0}" width="{cw}" height="{ch}" rx="6" fill="{C["bg1"]}" stroke="{C["cyan"]}" stroke-opacity=".5"/>'
    fill_x, fill_w = cx0 + 10, cw - 20
    svg += (
        f'<rect x="{fill_x}" y="{cy0+90}" width="{fill_w}" height="70" fill="{C["cyan"]}" opacity=".18">'
        f'<animate attributeName="y" values="{cy0+90};{cy0+70};{cy0+90}" dur="5s" repeatCount="indefinite"/>'
        f'<animate attributeName="height" values="70;90;70" dur="5s" repeatCount="indefinite"/></rect>'
    )
    svg += (
        f'<line x1="{fill_x}" y1="{cy0+90}" x2="{fill_x+fill_w}" y2="{cy0+90}" stroke="{C["cyan"]}" stroke-width="1.6">'
        f'<animate attributeName="y1" values="{cy0+90};{cy0+70};{cy0+90}" dur="5s" repeatCount="indefinite"/>'
        f'<animate attributeName="y2" values="{cy0+90};{cy0+70};{cy0+90}" dur="5s" repeatCount="indefinite"/></line>'
    )
    svg += (
        f'<g><circle cx="{cx0+24}" cy="{cy0+22}" r="4" fill="{C["cyan3"]}">'
        f'<animate attributeName="opacity" values="1;.25;1" dur="1.1s" repeatCount="indefinite"/></circle>'
        f'<text x="{cx0+34}" y="{cy0+26}" font-family="{MONO}" font-size="10" fill="{C["cyan3"]}">AHT10 · I2C</text></g>'
    )
    rx, ry = 330, 60
    svg += f'<text x="{rx}" y="{ry}" font-family="{MONO}" font-size="10" letter-spacing="2" fill="{C["dim"]}">TEMP</text>'
    svg += "".join(
        f'<text x="{rx}" y="{ry+26}" font-family="{MONO}" font-size="22" font-weight="700" fill="{C["ink"]}" filter="url(#glowS)" opacity="0">{t}'
        f'<animate attributeName="opacity" dur="4.8s" repeatCount="indefinite" keyTimes="{kt}" values="{vals}"/></text>'
        for t, (kt, vals) in ((tt, window(k, 1, 4, 4)) for k, tt in enumerate(["24.1°C", "24.4°C", "24.6°C", "24.3°C"]))
    )
    svg += f'<text x="{rx}" y="{ry+46}" font-family="{MONO}" font-size="10" letter-spacing="2" fill="{C["dim"]}">HUMIDITY</text>'
    svg += "".join(
        f'<text x="{rx}" y="{ry+72}" font-family="{MONO}" font-size="22" font-weight="700" fill="{C["cyan"]}" filter="url(#glowS)" opacity="0">{t}'
        f'<animate attributeName="opacity" dur="5.2s" repeatCount="indefinite" keyTimes="{kt}" values="{vals}"/></text>'
        for t, (kt, vals) in ((tt, window(k, 1, 4, 4)) for k, tt in enumerate(["58% RH", "61% RH", "63% RH", "60% RH"]))
    )
    gx, gy, gr = rx + 40, 176, 34
    svg += f'<path d="M{gx-gr} {gy} A{gr} {gr} 0 0 1 {gx+gr} {gy}" fill="none" stroke="{C["cyan"]}" stroke-opacity=".3" stroke-width="4"/>'
    svg += (
        f'<g transform="translate({gx} {gy})"><animateTransform attributeName="transform" type="rotate" '
        f'values="-40;40;-40" keyTimes="0;.5;1" calcMode="spline" keySplines="0.4 0 0.2 1;0.4 0 0.2 1" dur="6s" additive="sum" repeatCount="indefinite"/>'
        f'<line x1="0" y1="0" x2="0" y2="-{gr-4}" stroke="{C["cyan3"]}" stroke-width="2.5"/></g>'
    )
    svg += f'<text x="20" y="{H-16}" font-family="{MONO}" font-size="10.5" fill="{C["dim"]}">Raspberry Pi · Django · PostgreSQL · PCB</text>'
    svg += "</g>"
    svg += card_frame(W, H, cut=18)
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-label="Animated environmental control system dashboard">{svg}</svg>\n'


def sign_translator_svg():
    W, H = 480, 260
    hx, hy = 150, 150
    finger_defs = [
        ((-46, -8), (-60, -46), (-64, -78)),
        ((-22, -30), (-24, -74), (-26, -108)),
        ((2, -34), (2, -82), (2, -118)),
        ((26, -30), (30, -74), (33, -106)),
        ((48, -14), (58, -46), (66, -70)),
    ]
    svg = _project_chrome(W, H, "SIGN LANGUAGE TRANSLATOR", "HAND.TRACK")
    palm = f'<polygon points="{hx-46},{hy-8} {hx-30},{hy+38} {hx+30},{hy+38} {hx+48},{hy-14} {hx+26},{hy-30} {hx-22},{hy-30}" fill="{C["cyan"]}" opacity=".08" stroke="{C["cyan"]}" stroke-opacity=".4"/>'
    svg += palm
    dot_i = 0
    for (j1, j2, j3) in finger_defs:
        x1, y1 = hx + j1[0], hy + j1[1]
        x2, y2 = hx + j2[0], hy + j2[1]
        x3, y3 = hx + j3[0], hy + j3[1]
        svg += f'<line x1="{hx}" y1="{hy-8}" x2="{x1}" y2="{y1}" stroke="{C["cyan"]}" stroke-opacity=".5"/>'
        svg += f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{C["cyan"]}" stroke-opacity=".5"/>'
        svg += f'<line x1="{x2}" y1="{y2}" x2="{x3}" y2="{y3}" stroke="{C["cyan"]}" stroke-opacity=".5"/>'
        for (x, y) in ((x1, y1), (x2, y2), (x3, y3)):
            delay = dot_i * 0.12
            svg += (
                f'<circle cx="{x}" cy="{y}" r="3.2" fill="{C["cyan3"]}" filter="url(#glowS)">'
                f'<animate attributeName="opacity" values=".35;1;.35" dur="1.6s" begin="{num(delay,2)}s" repeatCount="indefinite"/></circle>'
            )
            dot_i += 1
    svg += f'<circle cx="{hx}" cy="{hy-8}" r="4" fill="{C["cyan3"]}" filter="url(#glowS)"/>'

    tx0, ty0 = 300, 70
    svg += f'<text x="{tx0}" y="{ty0}" font-family="{MONO}" font-size="11" fill="{C["dim"]}">&gt; translating sign stream_</text>'
    word = "HELLO"
    T = 6.0
    letters = ""
    for i, ch in enumerate(word):
        a = (i * 0.4) / T
        letters += (
            f'<text x="{tx0 + i*22}" y="{ty0+40}" font-family="{MONO}" font-size="26" font-weight="700" fill="{C["cyan"]}" opacity="0" filter="url(#glowS)">{ch}'
            f'<animate attributeName="opacity" dur="{num(T)}s" repeatCount="indefinite" '
            f'keyTimes="0;{num(a,4)};{num(a+0.001,4)};0.85;1" values="0;0;1;1;0"/></text>'
        )
    svg += letters
    svg += (
        f'<rect x="{tx0 + len(word)*22 + 4}" y="{ty0+18}" width="10" height="24" fill="{C["cyan3"]}">'
        f'<animate attributeName="opacity" values="1;1;0;0" keyTimes="0;.5;.5001;1" dur=".9s" repeatCount="indefinite"/></rect>'
    )
    svg += f'<line x1="{tx0}" y1="{ty0+58}" x2="{W-24}" y2="{ty0+58}" stroke="{C["cyan"]}" stroke-opacity=".25"/>'
    svg += f'<text x="{tx0}" y="{ty0+80}" font-family="{MONO}" font-size="10.5" fill="{C["dim"]}">landmarks · classifier</text>'
    svg += f'<text x="{tx0}" y="{ty0+96}" font-family="{MONO}" font-size="10.5" fill="{C["dim"]}">text-to-speech output</text>'
    svg += f'<text x="20" y="{H-16}" font-family="{MONO}" font-size="10.5" fill="{C["dim"]}">Python · OpenCV · MediaPipe · Machine Learning</text>'
    svg += "</g>"
    svg += card_frame(W, H, cut=18)
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-label="Animated real-time sign language translator">{svg}</svg>\n'


# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    icons = json.loads((HERE / "icons.json").read_text())
    (OUT / "tech-particles.svg").write_text(particles_svg(icons), encoding="utf-8")
    (OUT / "header.svg").write_text(header_svg(), encoding="utf-8")
    (OUT / "skills.svg").write_text(skills_svg(), encoding="utf-8")
    (OUT / "divider.svg").write_text(divider_svg(), encoding="utf-8")
    (OUT / "about.svg").write_text(about_svg(), encoding="utf-8")
    (OUT / "education.svg").write_text(education_svg(), encoding="utf-8")
    (OUT / "awards.svg").write_text(awards_svg(), encoding="utf-8")
    (OUT / "telemetry.svg").write_text(telemetry_svg(), encoding="utf-8")
    (OUT / "project-robotic-arm.svg").write_text(robotic_arm_svg(), encoding="utf-8")
    (OUT / "project-humidity.svg").write_text(humidity_svg(), encoding="utf-8")
    (OUT / "project-sign-translator.svg").write_text(sign_translator_svg(), encoding="utf-8")
    (OUT / "title-tech.svg").write_text(section_title_svg("TECH.STACK"), encoding="utf-8")
    (OUT / "title-projects.svg").write_text(section_title_svg("NOTABLE.PROJECTS"), encoding="utf-8")
    (OUT / "title-telemetry.svg").write_text(section_title_svg("GITHUB.TELEMETRY"), encoding="utf-8")
    (OUT / "title-contact.svg").write_text(section_title_svg("CONTACT", tag="OPEN"), encoding="utf-8")
    for f in sorted(OUT.glob("*.svg")):
        print(f"{f.name:28s}{f.stat().st_size/1024:8.1f} KB")
