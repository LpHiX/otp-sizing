import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, replace
from typing import Callable, Optional


from otp.pump.sizing import size_pump
from otp.pump.analysis import Pump

@dataclass
class Panel:
    ylabel: str
    xlabel: str
    x: Callable                        # perf -> x
    y: Callable                        # perf -> y
    ref:  Optional[Callable] = None    # perf -> horizontal reference, drawn per curve (e.g. NPSHa)
    mask: Optional[Callable] = None    # perf -> keep?  False points become gaps (e.g. post-cutoff)
    secondary: Optional[tuple] = None  # (label, fwd, inv) right-hand axis

# ── shared config: change these once, all three functions follow ────────────
_LINESTYLES = {"lock": "-", "barske": "--", "barske_simple": ":"}
_CMAP       = "viridis"
_DESIGN_KW  = dict(marker="*", ms=16, color="red", mec="k", zorder=5)
_REF_KW     = dict(ls=":", lw=2)

# ── protection layer: the only places that touch size_pump / point_performance
def _safe(fn, *a, default=np.nan, **k):
    try:    return fn(*a, **k)
    except Exception: return default

def _build_pump(req, choices, sizing_method):
    """size_pump can fail on infeasible geometry -> return None (curve becomes a gap)."""
    try:    return Pump(req, choices, size_pump(req, choices, sizing_method=sizing_method))
    except Exception: return None

def _eval_point(pump, **kw):
    """One operating point; None if geometry missing or the solve throws."""
    if pump is None: return None
    try:    return pump.point_performance(**kw)
    except Exception: return None

def _eval_curve(pump, Qs, base_kw):
    return [_eval_point(pump, Q=Q, **base_kw) for Q in Qs]

# ── panel evaluation: single source of truth for x/y/mask/ref + NaN gaps ─────
def _series(panel, perfs):
    xs, ys = [], []
    for p in perfs:
        if p is None:                       # failed point -> gap
            xs.append(np.nan); ys.append(np.nan); continue
        keep = panel.mask is None or bool(_safe(panel.mask, p, default=False))
        xs.append(_safe(panel.x, p))
        ys.append(_safe(panel.y, p) if keep else np.nan)
    return np.asarray(xs, float), np.asarray(ys, float)

def _scalar_grid(G, fn):                    # perf-grid -> float grid (for maps)
    return np.array([[_safe(fn, p) if p is not None else np.nan for p in row]
                     for row in G], float)

def _plot_curve(ax, panel, perfs, color, ls="-", label=None, draw_ref=True):
    xs, ys = _series(panel, perfs)
    ax.plot(xs, ys, ls, color=color, label=label)
    if draw_ref and panel.ref is not None:
        p0 = next((p for p in perfs if p is not None), None)
        r  = None if p0 is None else _safe(panel.ref, p0, default=None)
        if r is not None: ax.axhline(r, color=color, **_REF_KW)

# ── axes scaffolding ────────────────────────────────────────────────────────
def _make_axes(n, ncols, figsize):
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize or (5*ncols, 4*nrows))
    return fig, np.asarray(axes).reshape(-1).tolist()

def _label_axes(panels, axs):
    for panel, ax in zip(panels, axs):
        ax.set_xlabel(panel.xlabel); ax.set_ylabel(panel.ylabel); ax.grid(True)
        if panel.secondary:
            lbl, fwd, inv = panel.secondary
            ax.secondary_yaxis("right", functions=(fwd, inv)).set_ylabel(lbl)
    for ax in axs[len(panels):]: ax.set_visible(False)

def _stars(panels, axs, design):
    ds = [] if design is None else (list(design) if isinstance(design, (list, tuple)) else [design])
    for panel, ax in zip(panels, axs):
        for j, d in enumerate(ds):
            ax.plot(_safe(panel.x, d), _safe(panel.y, d),
                    label="design" if j == 0 else None, **_DESIGN_KW)

def _legend(fig, axs, values, title):
    h, l = axs[0].get_legend_handles_labels()
    if h:
        fig.legend(h, l, loc="lower center", ncol=max(1, len(values)),
                   title=title, bbox_to_anchor=(0.5, -0.04), fontsize="small")
    fig.tight_layout()


# ════════════════════════════════════════════════════════════════════════════
def sweep_plot(pump, panels, Qs, sweep, fixed, methods=("lock",),
               design=None, ncols=4, figsize=None, cmap=_CMAP):
    """Sweep an OPERATING kwarg (rpm, p_upstream, …) on a FIXED geometry."""
    name, values = sweep
    fig, axs = _make_axes(len(panels), ncols, figsize)
    colors = plt.get_cmap(cmap)(np.linspace(0, 0.85, len(values)))
    for vi, val in enumerate(values):
        for mi, method in enumerate(methods):
            perfs = _eval_curve(pump, Qs, {**fixed, "method": method, name: val})
            for panel, ax in zip(panels, axs):
                _plot_curve(ax, panel, perfs, colors[vi], _LINESTYLES.get(method, "-"),
                            label=f"{name} {val}" if mi == 0 else None, draw_ref=(mi == 0))
    _label_axes(panels, axs); _stars(panels, axs, design); _legend(fig, axs, values, name)
    return fig


def design_sweep_plot(req, base_choices, panels, Qs, sweep, fixed,
                      method="lock", sizing_method="lock",
                      design=None, ncols=4, figsize=None, cmap=_CMAP):
    """Sweep a GEOMETRY-changing choice (n_blades, v_inlet, …): re-size per value."""
    name, values = sweep
    fig, axs = _make_axes(len(panels), ncols, figsize)
    colors = plt.get_cmap(cmap)(np.linspace(0, 0.85, len(values)))
    for vi, val in enumerate(values):
        ch    = replace(base_choices, **{name: val})
        pump  = _build_pump(req, ch, sizing_method)          # None -> curve is a clean gap
        perfs = _eval_curve(pump, Qs, {**fixed, "rpm": ch.rpm, "method": method})
        for panel, ax in zip(panels, axs):
            _plot_curve(ax, panel, perfs, colors[vi], _LINESTYLES.get(method, "-"),
                        label=f"{name} {val}")
    _label_axes(panels, axs); _stars(panels, axs, design); _legend(fig, axs, values, name)
    return fig


def design_map(req, base_choices, panels, x, y, fixed, feasible=None,
               method="lock", sizing_method="lock", at_Q=None,
               ncols=4, figsize=None, cmap=_CMAP, levels=14):
    """2-D map over two design choices. One contour subplot per panel (panel.y = z,
    panel.ylabel = title). Cells where sizing/solve fail stay blank. Optional
    `feasible(perf)` (>=0 good) draws the boundary, hatches the bad region, and
    restricts each panel's ★ (max of panel.y among feasible cells)."""
    (xn, xv), (yn, yv) = x, y
    xv, yv = np.asarray(xv, float), np.asarray(yv, float)

    # build the perf grid ONCE (shared by every panel) — protected per cell
    G = np.empty((len(yv), len(xv)), dtype=object)
    for j, yy in enumerate(yv):
        for i, xx in enumerate(xv):
            ch   = replace(base_choices, **{xn: float(xx), yn: float(yy)})
            pump = _build_pump(req, ch, sizing_method)
            G[j, i] = _eval_point(pump, Q=(at_Q if at_Q is not None else req.Q_req),
                                  rpm=ch.rpm, method=method, **fixed)

    X, Y = np.meshgrid(xv, yv)
    Fz   = _scalar_grid(G, feasible) if feasible else None
    fig, axs = _make_axes(len(panels), ncols, figsize)
    for panel, ax in zip(panels, axs):
        Z = _scalar_grid(G, panel.y)
        if np.isfinite(Z).any():
            fig.colorbar(ax.contourf(X, Y, Z, levels=levels, cmap=cmap), ax=ax)
        if Fz is not None and np.isfinite(Fz).any():
            ax.contour(X, Y, Fz, levels=[0], colors="r", linewidths=2)
            ax.contourf(X, Y, np.where(np.isfinite(Fz) & (Fz < 0), 1.0, 0.0),
                        levels=[0.5, 1.5], colors="none", hatches=["xx"])
        Zf = np.where((Fz >= 0) if Fz is not None else np.isfinite(Z), Z, -np.inf)
        if np.isfinite(Zf).any() and Zf.max() > -np.inf:
            j, i = np.unravel_index(np.nanargmax(Zf), Zf.shape)
            ax.plot(xv[i], yv[j], **_DESIGN_KW)
        ax.set_title(panel.ylabel); ax.set_xlabel(xn); ax.set_ylabel(yn)
    for ax in axs[len(panels):]: ax.set_visible(False)
    fig.tight_layout()
    return fig
