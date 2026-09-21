#!/usr/bin/env python3
"""
Genera los SVG animados (SMIL, sin JavaScript) para el README de GitHub.

Uso:
    pip install cairosvg numpy scipy
    python tools/build_assets.py

Edita las secciones CONFIG de abajo para cambiar tecnologías, habilidades o colores.
Las rutas de los logos vienen de tools/icons.json (paths de simple-icons.org).
"""
import io
import json
import math
import os
import random
from pathlib import Path

import cairosvg
import numpy as np
from PIL import Image
from scipy.cluster.vq import kmeans2
from scipy.ndimage import binary_erosion
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).parent
OUT = Path(os.environ.get("OUT_DIR", HERE / "assets"))
OUT.mkdir(parents=True, exist_ok=True)

# ───────────────────────── CONFIG ─────────────────────────
PALETTE = {
    "bg0": "#07070d",
    "bg1": "#0d0b1f",
    "cyan": "#00f0ff",
    "magenta": "#ff2a6d",
    "yellow": "#fcee0a",
    "violet": "#b026ff",
    "mint": "#05ffa1",
    "dim": "#56628a",
    "ink": "#e6fbff",
}
NEON_CYCLE = ["cyan", "magenta", "yellow", "violet", "mint"]

MONO = "'JetBrains Mono','Fira Code','SFMono-Regular',Consolas,'Courier New',monospace"

# Radar (dominios) — valores 0-100. AJUSTA A TU NIVEL REAL.
RADAR = [
    ("EMBEDDED", 82),
    ("PYTHON", 90),
    ("DATA / BI", 72),
    ("COMPUTER VISION", 70),
    ("RPA / AUTOMATION", 78),
    ("WEB BACKEND", 70),
    ("CLOUD / DEVOPS", 58),
    ("IC / SEMICONDUCTORS", 64),
]
# Barras (herramientas) — valores 0-100. AJUSTA A TU NIVEL REAL.
BARS = [
    ("Python", 90),
    ("SQL", 75),
    ("UiPath", 78),
    ("OpenCV", 72),
    ("Django", 70),
    (".NET", 66),
    ("Java", 64),
    ("Docker", 58),
]

HEADER_LINES = [
    "Electronics Engineer",
    "Teaching Assistant",
    "Software Developer",
    "IC & Sensor Enthusiast",
    "Data Analyst in the Making",
]
NAME = "ABIMAEL FRANCO"
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
#  1) PARTÍCULAS QUE SE TRANSFORMAN EN LOGOS
# ═══════════════════════════════════════════════════════════
def logo_mask(path_d, res=260, invert=False):
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="{res}" height="{res}">'
        f'<rect width="24" height="24" fill="black"/><path d="{path_d}" fill="white"/></svg>'
    )
    png = cairosvg.svg2png(bytestring=svg.encode())
    img = np.array(Image.open(io.BytesIO(png)).convert("L"))
    m = img > 128
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
    # Rellenar si kmeans devolvió menos de n
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

    # Emparejar cada logo con el siguiente para que el movimiento sea corto y fluido
    order = [np.arange(n)]
    aligned = [sets[0]]
    for k in range(1, N):
        prev = aligned[-1]
        d = np.linalg.norm(prev[:, None, :] - sets[k][None, :, :], axis=2)
        r, c = linear_sum_assignment(d)
        aligned.append(sets[k][c])
    pos = np.stack(aligned)  # (N, n, 2)

    colors = [C[NEON_CYCLE[k % len(NEON_CYCLE)]] for k in range(N)]

    # Color global animado (grupo)
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
            # hold hasta s
            splines.append("0 0 1 1")
            times.append(s / T); xs.append(a[0]); ys.append(a[1])
            splines.append("0.5 0 1 1")
            times.append(mid_t / T); xs.append(mid[0]); ys.append(mid[1])
            splines.append("0 0 0.5 1")
            times.append(e / T); xs.append(b[0]); ys.append(b[1])
        if times[-1] < 1.0:
            splines.append("0 0 1 1")
            times.append(1.0); xs.append(pos[0, i, 0]); ys.append(pos[0, i, 1])
        # Monotonía numérica
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
    """keyTimes/values para que un elemento sea visible sólo durante el turno k."""
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
    # kt tiene un elemento extra al principio -> ajustar
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
        # barra de progreso del módulo activo
        a, b = k * S / T, (k + 1) * S / T
        if k == 0:
            bkt = [0.0, b, b + 0.0004, 1.0]; bv = ["96", "96", "0", "0"]
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
  <clipPath id="card"><polygon points="0,0 {W-26},0 {W},26 {W},{H} 26,{H} 0,{H-26}"/></clipPath>
</defs>

<g clip-path="url(#card)">
  <rect width="{W}" height="{H}" fill="url(#bg)"/>
  <rect width="{W}" height="{H}" fill="url(#grid)"/>
  <circle cx="{cx}" cy="{cy}" r="150" fill="{C["cyan"]}" opacity=".035"/>

  <!-- anillos giratorios -->
  <g fill="none" stroke-width="1.2">
    <circle cx="{cx}" cy="{cy}" r="122" stroke="{C["cyan"]}" stroke-opacity=".45" stroke-dasharray="46 22 6 22">
      <animateTransform attributeName="transform" type="rotate" from="0 {cx} {cy}" to="360 {cx} {cy}" dur="28s" repeatCount="indefinite"/></circle>
    <circle cx="{cx}" cy="{cy}" r="134" stroke="{C["magenta"]}" stroke-opacity=".4" stroke-dasharray="4 14 60 14">
      <animateTransform attributeName="transform" type="rotate" from="360 {cx} {cy}" to="0 {cx} {cy}" dur="40s" repeatCount="indefinite"/></circle>
    <circle cx="{cx}" cy="{cy}" r="146" stroke="{C["dim"]}" stroke-opacity=".35" stroke-dasharray="1 7"/>
  </g>

  <!-- partículas -->
  <g fill="{C["cyan"]}" filter="url(#glow)">
    <animate attributeName="fill" dur="{dur}" repeatCount="indefinite" keyTimes="{ck}" values="{cv}"/>
    {"".join(P["circles"])}
  </g>

  {"".join(labels)}

  <!-- panel derecho -->
  <line x1="600" y1="34" x2="600" y2="{H-34}" stroke="{C["cyan"]}" stroke-opacity=".25"/>
  <text x="646" y="50" font-family="{MONO}" font-size="12" fill="{C["dim"]}" letter-spacing="2">$ ls ./stack --active</text>
  {"".join(rows)}

  <!-- HUD -->
  <text x="36" y="42" font-family="{MONO}" font-size="12" letter-spacing="3" fill="{C["cyan"]}" opacity=".85">// STACK.SCAN</text>
  <g font-family="{MONO}" font-size="11" letter-spacing="2" fill="{C["mint"]}" text-anchor="end">
    <circle cx="{W-150}" cy="38" r="3.5"><animate attributeName="opacity" values="1;.15;1" dur="1.4s" repeatCount="indefinite"/></circle>
    <text x="{W-40}" y="42">SYS.ONLINE</text>
  </g>
  <text x="36" y="{H-22}" font-family="{MONO}" font-size="10" letter-spacing="2" fill="{C["dim"]}">PARTICLES {P["n"]} · MORPH LOOP {num(T)}s</text>

  <!-- scanlines + haz de escaneo -->
  <rect width="{W}" height="{H}" fill="url(#scan)"/>
  <rect x="0" y="-60" width="{W}" height="60" fill="url(#beam)">
    <animate attributeName="y" values="-60;{H}" dur="7s" repeatCount="indefinite"/></rect>
</g>

<!-- marco -->
<polygon points="1,1 {W-27},1 {W-1},27 {W-1},{H-1} 27,{H-1} 1,{H-27}" fill="none" stroke="{C["cyan"]}" stroke-opacity=".5" stroke-width="1.2"/>
<path d="M1 60V1h60" fill="none" stroke="{C["cyan"]}" stroke-width="2.5"/>
<path d="M{W-1} {H-60}V{H-1}H{W-61}" fill="none" stroke="{C["magenta"]}" stroke-width="2.5"/>
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

    # ─ typing
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
        # cursor: sigue el borde del texto que se está escribiendo
        for t, xv in ((a, x0), (a + 1.5, x0 + wtxt), (a + 2.9, x0 + wtxt), (a + 3.4, x0)):
            cursor_kt.append(min(t / T, 1.0)); cursor_x.append(xv)
    cursor_kt[-1] = 1.0
    for j in range(1, len(cursor_kt)):
        cursor_kt[j] = max(cursor_kt[j], cursor_kt[j - 1])
    assert len(cursor_kt) == len(cursor_x)
    check_times(cursor_kt)

    # ─ glitch en el nombre
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

    # ─ circuitos decorativos (derecha)
    traces = [
        ("M700 60H790L820 90H960", C["cyan"], 0),
        ("M760 250H850L880 220H960V170", C["magenta"], 1.3),
        ("M660 30H730L750 50H940", C["violet"], 2.4),
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
        for x, y, c in [(960, 90, C["cyan"]), (960, 170, C["magenta"]), (940, 50, C["violet"])]
    )

    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-label="Abimael Franco — Electronics Engineer, Software Developer, Data Analyst in the Making">
<title>Abimael Franco</title>
<defs>
  <filter id="glow" x="-10%" y="-40%" width="120%" height="180%"><feGaussianBlur stdDeviation="4" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
  <filter id="glowS" x="-20%" y="-50%" width="140%" height="200%"><feGaussianBlur stdDeviation="2" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
  <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="{C["bg0"]}"/><stop offset=".6" stop-color="{C["bg1"]}"/><stop offset="1" stop-color="#1a0a2e"/></linearGradient>
  <linearGradient id="rule" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="{C["magenta"]}"/><stop offset=".5" stop-color="{C["cyan"]}"/><stop offset="1" stop-color="{C["violet"]}"/></linearGradient>
  <pattern id="grid" width="32" height="32" patternUnits="userSpaceOnUse"><path d="M32 0H0V32" fill="none" stroke="{C["cyan"]}" stroke-opacity=".05"/></pattern>
  <pattern id="scan" width="4" height="4" patternUnits="userSpaceOnUse"><rect width="4" height="1" fill="#000" opacity=".28"/></pattern>
  <linearGradient id="beam" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="{C["magenta"]}" stop-opacity="0"/><stop offset=".5" stop-color="{C["magenta"]}" stop-opacity=".14"/><stop offset="1" stop-color="{C["magenta"]}" stop-opacity="0"/></linearGradient>
  <clipPath id="card"><polygon points="0,0 {W-30},0 {W},30 {W},{H} 30,{H} 0,{H-30}"/></clipPath>
  {"".join(clips)}
</defs>

<g clip-path="url(#card)">
  <rect width="{W}" height="{H}" fill="url(#bg)"/>
  <rect width="{W}" height="{H}" fill="url(#grid)"/>
  {tr}{nodes}

  <text x="{nx}" y="52" font-family="{MONO}" font-size="12" letter-spacing="3" fill="{C["mint"]}" opacity=".9">&gt; whoami<tspan fill="{C["dim"]}"> --profile --verbose</tspan></text>

  <g font-family="'Arial Black','Impact','Segoe UI',Arial,sans-serif">
    {glitch_layer(C["magenta"], gx_m)}
    {glitch_layer(C["cyan"], gx_c)}
    <text x="{nx}" y="150" font-size="76" font-weight="800" fill="{C["ink"]}" textLength="800" lengthAdjust="spacingAndGlyphs" filter="url(#glow)">{NAME}</text>
  </g>

  <rect x="{nx}" y="172" width="800" height="3" fill="url(#rule)" filter="url(#glowS)"/>

  <g font-family="{MONO}">
    <text x="{nx}" y="{y_line}" font-size="{fs}" fill="{C["magenta"]}" font-weight="700">&gt;</text>
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

<polygon points="1,1 {W-31},1 {W-1},31 {W-1},{H-1} 31,{H-1} 1,{H-31}" fill="none" stroke="{C["cyan"]}" stroke-opacity=".5" stroke-width="1.2"/>
<path d="M1 70V1h70" fill="none" stroke="{C["magenta"]}" stroke-width="2.5"/>
<path d="M{W-1} {H-70}V{H-1}H{W-71}" fill="none" stroke="{C["cyan"]}" stroke-width="2.5"/>
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
        f'<polygon points="{center_pts}" fill="{C["magenta"]}" fill-opacity=".2" stroke="{C["magenta"]}" stroke-width="2" stroke-linejoin="round" filter="url(#glowS)">'
        f'<animate attributeName="points" dur="{T}s" repeatCount="indefinite" calcMode="spline" keyTimes="{kts}" keySplines="{spl}" values="{center_pts};{full_pts};{full_pts};{center_pts}"/></polygon>'
    )
    dots = ""
    for i, (_, v) in enumerate(RADAR):
        x, y = pt(i, v)
        dots += (
            f'<circle cx="{cx}" cy="{cy}" r="3.4" fill="{C["yellow"]}" filter="url(#glowS)">'
            f'<animate attributeName="cx" dur="{T}s" repeatCount="indefinite" calcMode="spline" keyTimes="{kts}" keySplines="{spl}" values="{cx};{num(x)};{num(x)};{cx}"/>'
            f'<animate attributeName="cy" dur="{T}s" repeatCount="indefinite" calcMode="spline" keyTimes="{kts}" keySplines="{spl}" values="{cy};{num(y)};{num(y)};{cy}"/></circle>'
        )

    # barras
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
            f'<text x="{bx+bw+12}" y="{y+4}" font-size="12" fill="{C["yellow"]}" opacity="0">{v}%'
            f'<animate attributeName="opacity" dur="{T}s" repeatCount="indefinite" keyTimes="{bk}" values="0;0;1;1;0"/></text>'
            f"</g>"
        )

    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-label="Skill radar and tool proficiency bars">
<title>Skill matrix</title>
<defs>
  <filter id="glowS" x="-20%" y="-50%" width="140%" height="200%"><feGaussianBlur stdDeviation="2" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
  <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="{C["bg0"]}"/><stop offset="1" stop-color="{C["bg1"]}"/></linearGradient>
  <linearGradient id="bar" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="{C["cyan"]}"/><stop offset="1" stop-color="{C["magenta"]}"/></linearGradient>
  <pattern id="grid" width="28" height="28" patternUnits="userSpaceOnUse"><path d="M28 0H0V28" fill="none" stroke="{C["cyan"]}" stroke-opacity=".05"/></pattern>
  <pattern id="scan" width="4" height="4" patternUnits="userSpaceOnUse"><rect width="4" height="1" fill="#000" opacity=".28"/></pattern>
  <clipPath id="card"><polygon points="0,0 {W-26},0 {W},26 {W},{H} 26,{H} 0,{H-26}"/></clipPath>
</defs>
<g clip-path="url(#card)" font-family="{MONO}">
  <rect width="{W}" height="{H}" fill="url(#bg)"/>
  <rect width="{W}" height="{H}" fill="url(#grid)"/>
  <text x="36" y="42" font-size="12" letter-spacing="3" fill="{C["cyan"]}" opacity=".85">// SKILL.MATRIX</text>
  <text x="520" y="42" font-size="12" letter-spacing="3" fill="{C["magenta"]}" opacity=".9">// TOOLCHAIN.LOAD</text>
  <line x1="500" y1="34" x2="500" y2="{H-34}" stroke="{C["cyan"]}" stroke-opacity=".22"/>
  {rings}{axes}{poly}{dots}{labels}
  {bars}
  <rect width="{W}" height="{H}" fill="url(#scan)"/>
</g>
<polygon points="1,1 {W-27},1 {W-1},27 {W-1},{H-1} 27,{H-1} 1,{H-27}" fill="none" stroke="{C["cyan"]}" stroke-opacity=".5" stroke-width="1.2"/>
<path d="M1 60V1h60" fill="none" stroke="{C["magenta"]}" stroke-width="2.5"/>
<path d="M{W-1} {H-60}V{H-1}H{W-61}" fill="none" stroke="{C["cyan"]}" stroke-width="2.5"/>
</svg>
'''


# ═══════════════════════════════════════════════════════════
#  4) DIVISOR
# ═══════════════════════════════════════════════════════════
def divider_svg():
    W, H = 1000, 24
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="separator" aria-hidden="true">
<defs>
  <linearGradient id="l" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="{C["magenta"]}" stop-opacity="0"/><stop offset=".25" stop-color="{C["magenta"]}"/><stop offset=".5" stop-color="{C["cyan"]}"/><stop offset=".75" stop-color="{C["violet"]}"/><stop offset="1" stop-color="{C["violet"]}" stop-opacity="0"/></linearGradient>
  <linearGradient id="g" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="#fff" stop-opacity="0"/><stop offset=".5" stop-color="#fff"/><stop offset="1" stop-color="#fff" stop-opacity="0"/></linearGradient>
  <filter id="glow" x="-10%" y="-300%" width="120%" height="700%"><feGaussianBlur stdDeviation="2.5" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
</defs>
<rect x="60" y="11" width="{W-120}" height="2" fill="url(#l)" filter="url(#glow)"/>
<rect x="-160" y="10" width="160" height="4" fill="url(#g)" opacity=".9"><animate attributeName="x" values="-160;{W}" dur="3.6s" repeatCount="indefinite"/></rect>
<g transform="translate(500 12) rotate(45)"><rect x="-5" y="-5" width="10" height="10" fill="{C["bg0"]}" stroke="{C["yellow"]}" stroke-width="1.5"/></g>
<circle cx="500" cy="12" r="1.8" fill="{C["yellow"]}"><animate attributeName="opacity" values="1;.2;1" dur="1.6s" repeatCount="indefinite"/></circle>
</svg>
'''


# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    icons = json.loads((HERE / "icons.json").read_text())
    (OUT / "tech-particles.svg").write_text(particles_svg(icons), encoding="utf-8")
    (OUT / "header.svg").write_text(header_svg(), encoding="utf-8")
    (OUT / "skills.svg").write_text(skills_svg(), encoding="utf-8")
    (OUT / "divider.svg").write_text(divider_svg(), encoding="utf-8")
    for f in sorted(OUT.glob("*.svg")):
        print(f"{f.name:22s}{f.stat().st_size/1024:8.1f} KB")
