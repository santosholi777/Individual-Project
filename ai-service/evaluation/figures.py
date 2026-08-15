"""SVG figures for the evaluation report.

Written as hand-built SVG rather than through a plotting library: it adds no
dependency, and the output is a vector file that drops straight into a Word
document or a slide without going fuzzy.

These figures deliberately commit to a light appearance — they are printed
artefacts for a report, not UI that follows a viewer's theme.

Two figures, each chosen for its job:

* **Score distribution** — two overlaid histograms. The job is telling two
  classes apart, so this is the one categorical use here: two hues, in the fixed
  palette order, both direct-labelled and carried by a legend, never colour
  alone. The threshold line is what makes it readable as a decision.
* **Threshold trade-off** — three rates as the threshold moves. Shows *why* a
  threshold was chosen rather than asserting it.
"""

from __future__ import annotations

import html
from pathlib import Path
from xml.etree import ElementTree

import numpy as np

from evaluation.metrics import ThresholdReport, VerificationMetrics

# Palette — validated against a white surface for contrast and CVD separation.
# Slots are assigned in fixed order and never cycled.
_SERIES_1 = "#2a78d6"  # slot 1, blue — genuine
_SERIES_2 = "#1baf7a"  # slot 2, aqua — impostor
_CRITICAL = "#d03b3b"
_WARNING = "#fab219"
_INK = "#10131a"
_INK_SECONDARY = "#4d5462"
_INK_MUTED = "#7a8394"
_GRID = "#e6e9ee"
_BASELINE = "#c8cdd6"
_SURFACE = "#ffffff"

#: Font stack for the figures. The quotes around "Segoe UI" MUST be XML-escaped:
#: raw double quotes would terminate the font-family attribute early and make the
#: whole SVG fail to parse.
_FONT = "system-ui, -apple-system, &quot;Segoe UI&quot;, Roboto, sans-serif"


def _escape(text: str) -> str:
    """Escape text for safe inclusion in SVG."""
    return html.escape(str(text), quote=True)


def _histogram(values: np.ndarray, bins: np.ndarray) -> np.ndarray:
    """Bin values, returning counts normalised to a 0–1 share."""
    if values.size == 0:
        return np.zeros(len(bins) - 1)
    counts, _ = np.histogram(values, bins=bins)
    total = counts.sum()
    return counts / total if total else counts.astype(float)


def score_distribution_svg(
    metrics: VerificationMetrics,
    threshold: float,
    width: int = 760,
    height: int = 340,
) -> str:
    """Render genuine vs impostor score distributions.

    The single most useful figure for the report: it shows *why* the threshold
    works, or that it cannot.

    Args:
        metrics: Collected genuine/impostor scores.
        threshold: Operating threshold, drawn as a vertical rule.
        width: Figure width in pixels.
        height: Figure height in pixels.

    Returns:
        A complete standalone SVG document.
    """
    pad = {"top": 46, "right": 24, "bottom": 52, "left": 56}
    plot_w = width - pad["left"] - pad["right"]
    plot_h = height - pad["top"] - pad["bottom"]

    bins = np.linspace(-0.4, 1.0, 36)
    genuine = _histogram(metrics.genuine, bins)
    impostor = _histogram(metrics.impostor, bins)
    peak = max(genuine.max(initial=0.0), impostor.max(initial=0.0), 0.001)

    def x_of(score: float) -> float:
        return pad["left"] + (score - bins[0]) / (bins[-1] - bins[0]) * plot_w

    def y_of(share: float) -> float:
        return pad["top"] + plot_h * (1 - share / peak)

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" font-family="{_FONT}">',
        f'<rect width="{width}" height="{height}" fill="{_SURFACE}"/>',
        f'<text x="{pad["left"]}" y="22" font-size="14" font-weight="650" '
        f'fill="{_INK}">Similarity score distribution</text>',
        f'<text x="{pad["left"]}" y="38" font-size="11" fill="{_INK_MUTED}">'
        f"Same person vs different people · {metrics.genuine.size} genuine, "
        f"{metrics.impostor.size} impostor comparisons</text>",
    ]

    # Gridlines: hairline, solid, recessive.
    for tick in np.linspace(0, peak, 4):
        y = y_of(float(tick))
        parts.append(
            f'<line x1="{pad["left"]}" x2="{width - pad["right"]}" y1="{y:.1f}" '
            f'y2="{y:.1f}" stroke="{_GRID}" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{pad["left"] - 8}" y="{y + 4:.1f}" font-size="10" '
            f'fill="{_INK_MUTED}" text-anchor="end">{tick * 100:.0f}%</text>'
        )

    # Bars: a 2px surface gap keeps neighbours distinct without a stroke.
    bar_w = plot_w / (len(bins) - 1)
    for series, colour in ((impostor, _SERIES_2), (genuine, _SERIES_1)):
        for index, share in enumerate(series):
            if share <= 0:
                continue
            x = x_of(float(bins[index]))
            y = y_of(float(share))
            parts.append(
                f'<rect x="{x + 1:.1f}" y="{y:.1f}" width="{max(bar_w - 2, 1):.1f}" '
                f'height="{pad["top"] + plot_h - y:.1f}" fill="{colour}" '
                f'fill-opacity="0.85" rx="2"/>'
            )

    # Axis.
    parts.append(
        f'<line x1="{pad["left"]}" x2="{width - pad["right"]}" '
        f'y1="{pad["top"] + plot_h}" y2="{pad["top"] + plot_h}" '
        f'stroke="{_BASELINE}" stroke-width="1"/>'
    )
    for tick in np.arange(-0.25, 1.01, 0.25):
        x = x_of(float(tick))
        parts.append(
            f'<text x="{x:.1f}" y="{pad["top"] + plot_h + 16}" font-size="10" '
            f'fill="{_INK_MUTED}" text-anchor="middle">{tick:.2f}</text>'
        )
    parts.append(
        f'<text x="{pad["left"] + plot_w / 2:.1f}" y="{height - 12}" font-size="11" '
        f'fill="{_INK_SECONDARY}" text-anchor="middle">Cosine similarity</text>'
    )

    # Threshold rule — the decision boundary the figure exists to explain.
    tx = x_of(threshold)
    parts.append(
        f'<line x1="{tx:.1f}" x2="{tx:.1f}" y1="{pad["top"] - 6}" '
        f'y2="{pad["top"] + plot_h}" stroke="{_CRITICAL}" stroke-width="2" '
        f'stroke-dasharray="4 3"/>'
    )
    anchor = "start" if threshold < 0.6 else "end"
    offset = 5 if threshold < 0.6 else -5
    parts.append(
        f'<text x="{tx + offset:.1f}" y="{pad["top"] - 10}" font-size="10" '
        f'font-weight="650" fill="{_CRITICAL}" text-anchor="{anchor}">'
        f"threshold {threshold:.2f}</text>"
    )

    # Legend lives in the header band, which is always free of marks — inside the
    # plot it would collide with whichever distribution happens to sit there.
    # Identity is never colour-alone, and the aqua slot is below 3:1 on white, so
    # these labels are required rather than a courtesy.
    legend = [("Same person (genuine)", _SERIES_1), ("Different people (impostor)", _SERIES_2)]
    lx = width - pad["right"] - 168
    for index, (label, colour) in enumerate(legend):
        ly = 18 + index * 15
        parts.append(
            f'<rect x="{lx}" y="{ly - 8}" width="9" height="9" rx="2" fill="{colour}"/>'
        )
        parts.append(
            f'<text x="{lx + 14}" y="{ly}" font-size="10.5" fill="{_INK_SECONDARY}">'
            f"{_escape(label)}</text>"
        )

    # Direct mean labels, clamped inside the plot so the text is never clipped
    # by the figure edge (a mean near 1.0 sits hard against the right margin).
    for series, mean in (
        (metrics.genuine, metrics.genuine_mean),
        (metrics.impostor, metrics.impostor_mean),
    ):
        if not series.size:
            continue
        label_x = min(max(x_of(mean), pad["left"] + 26), width - pad["right"] - 26)
        parts.append(
            f'<text x="{label_x:.1f}" y="{pad["top"] + 14}" font-size="10" '
            f'font-weight="650" fill="{_INK}" text-anchor="middle">'
            f"mean {mean:.2f}</text>"
        )

    parts.append("</svg>")
    return "\n".join(parts)


def threshold_sweep_svg(
    sweep: list[ThresholdReport],
    chosen: float,
    width: int = 760,
    height: int = 340,
) -> str:
    """Render how the three outcome rates trade off against the threshold.

    Args:
        sweep: Threshold reports across the range.
        chosen: The threshold in use, drawn as a vertical rule.
        width: Figure width in pixels.
        height: Figure height in pixels.

    Returns:
        A complete standalone SVG document.
    """
    # A deeper top margin than the distribution figure: the "in use" caption sits
    # above the plot and must clear the subtitle.
    pad = {"top": 58, "right": 122, "bottom": 52, "left": 56}
    plot_w = width - pad["left"] - pad["right"]
    plot_h = height - pad["top"] - pad["bottom"]

    def x_of(threshold: float) -> float:
        return pad["left"] + threshold * plot_w

    def y_of(rate: float) -> float:
        return pad["top"] + plot_h * (1 - rate)

    series = [
        ("Correct", _SERIES_1, [report.correct_rate for report in sweep]),
        ("Rejected", _WARNING, [report.rejection_rate for report in sweep]),
        ("Wrong ID", _CRITICAL, [report.wrong_id_rate for report in sweep]),
    ]
    if any(report.unknown_total for report in sweep):
        series.append(
            ("Stranger accepted", _INK_MUTED, [r.unknown_accept_rate for r in sweep])
        )

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" font-family="{_FONT}">',
        f'<rect width="{width}" height="{height}" fill="{_SURFACE}"/>',
        f'<text x="{pad["left"]}" y="22" font-size="14" font-weight="650" '
        f'fill="{_INK}">Threshold trade-off</text>',
        f'<text x="{pad["left"]}" y="38" font-size="11" fill="{_INK_MUTED}">'
        "How the outcome rates move as the recognition threshold changes</text>",
    ]

    for tick in np.linspace(0, 1, 5):
        y = y_of(float(tick))
        parts.append(
            f'<line x1="{pad["left"]}" x2="{width - pad["right"]}" y1="{y:.1f}" '
            f'y2="{y:.1f}" stroke="{_GRID}" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{pad["left"] - 8}" y="{y + 4:.1f}" font-size="10" '
            f'fill="{_INK_MUTED}" text-anchor="end">{tick * 100:.0f}%</text>'
        )

    for label, colour, rates in series:
        points = " ".join(
            f"{x_of(report.threshold):.1f},{y_of(rate):.1f}"
            for report, rate in zip(sweep, rates)
        )
        parts.append(
            f'<polyline points="{points}" fill="none" stroke="{colour}" '
            f'stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>'
        )

    # Direct labels at the line ends, which is why there is no legend box. Series
    # that finish at the same rate (commonly two of them at 0%) would print on
    # top of each other, so the labels are nudged apart vertically first.
    ends = sorted(
        ((y_of(rates[-1]), label, colour) for label, colour, rates in series),
        key=lambda item: item[0],
    )
    placed: list[tuple[float, str, str]] = []
    minimum_gap = 13.0
    for y, label, colour in ends:
        if placed and y - placed[-1][0] < minimum_gap:
            y = placed[-1][0] + minimum_gap
        placed.append((y, label, colour))

    for y, label, colour in placed:
        parts.append(
            f'<text x="{width - pad["right"] + 8}" y="{y + 4:.1f}" '
            f'font-size="10.5" font-weight="600" fill="{colour}">{_escape(label)}</text>'
        )

    parts.append(
        f'<line x1="{pad["left"]}" x2="{width - pad["right"]}" '
        f'y1="{pad["top"] + plot_h}" y2="{pad["top"] + plot_h}" '
        f'stroke="{_BASELINE}" stroke-width="1"/>'
    )
    for tick in np.arange(0, 1.01, 0.2):
        x = x_of(float(tick))
        parts.append(
            f'<text x="{x:.1f}" y="{pad["top"] + plot_h + 16}" font-size="10" '
            f'fill="{_INK_MUTED}" text-anchor="middle">{tick:.1f}</text>'
        )
    parts.append(
        f'<text x="{pad["left"] + plot_w / 2:.1f}" y="{height - 12}" font-size="11" '
        f'fill="{_INK_SECONDARY}" text-anchor="middle">Recognition threshold</text>'
    )

    cx = x_of(chosen)
    parts.append(
        f'<line x1="{cx:.1f}" x2="{cx:.1f}" y1="{pad["top"] - 6}" '
        f'y2="{pad["top"] + plot_h}" stroke="{_INK}" stroke-width="1.5" '
        f'stroke-dasharray="4 3" stroke-opacity="0.6"/>'
    )
    parts.append(
        f'<text x="{cx:.1f}" y="{pad["top"] - 10}" font-size="10" font-weight="650" '
        f'fill="{_INK}" text-anchor="middle">in use: {chosen:.2f}</text>'
    )

    parts.append("</svg>")
    return "\n".join(parts)


def _write_valid_svg(path: Path, markup: str) -> None:
    """Write an SVG, refusing to save one that is not well-formed XML.

    A malformed SVG fails silently — it writes fine and only shows itself as a
    broken image when someone opens the report. Parsing before writing turns
    that into a loud error at generation time.

    Raises:
        ValueError: If the markup is not well-formed XML.
    """
    try:
        ElementTree.fromstring(markup)
    except ElementTree.ParseError as exc:
        raise ValueError(
            f"Refusing to write malformed SVG to {path.name}: {exc}"
        ) from exc
    path.write_text(markup, encoding="utf-8")


def write_figures(
    directory: Path,
    metrics: VerificationMetrics,
    sweep: list[ThresholdReport],
    threshold: float,
) -> list[Path]:
    """Write both figures as SVG files.

    Args:
        directory: Destination directory (created if absent).
        metrics: Collected scores.
        sweep: Threshold sweep results.
        threshold: The operating threshold to mark.

    Returns:
        Paths of the written files.

    Raises:
        ValueError: If a figure would be malformed.
    """
    directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    distribution = directory / "score_distribution.svg"
    _write_valid_svg(distribution, score_distribution_svg(metrics, threshold))
    written.append(distribution)

    trade_off = directory / "threshold_tradeoff.svg"
    _write_valid_svg(trade_off, threshold_sweep_svg(sweep, threshold))
    written.append(trade_off)

    return written
