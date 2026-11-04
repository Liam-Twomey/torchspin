"""Interactive fitting panel for Jupyter (ipywidgets + matplotlib).

``FitSession`` runs :func:`torchspin.esfit` in a background thread and collects the
live progress (through ``FitOptions.progress``); ``FitPanel`` shows parameter sliders
with bounds and "fix" toggles, method/budget controls, Start/Stop/Reset buttons, and a
figure with data, current best fit, residual and the RMSD trace.  In JupyterLab use
``%matplotlib widget`` (ipympl) for a live canvas; with any other backend the figure is
refreshed as a PNG.

Requires ``ipywidgets`` and ``matplotlib`` (``pip install "torchspin[gui]"``).
"""
from __future__ import annotations

import io
import threading
import time
from dataclasses import replace
from typing import Callable, Optional, Sequence

import numpy as np

from torchspin.esfit import FitOptions, FitResult, ParameterHandler, esfit, _fit_scale_baseline_lsq

_METHODS = ['global', 'simplex', 'trf', 'levmar', 'lbfgsb', 'powell', 'swarm', 'genetic', 'montecarlo', 'grid']


def _require(module: str, purpose: str, extra: str = 'gui'):
    """Import an optional dependency, or explain how to install it.

    The panel's dependencies are deliberately optional, so the failure has to
    say which extra provides them rather than leaving a bare
    ``ModuleNotFoundError: No module named 'ipywidgets'``.
    """
    import importlib
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise ImportError(
            f"torchspin.fitgui needs {module} ({purpose}), which is not installed. "
            f'Install the optional dependencies with: pip install "torchspin[{extra}]"'
        ) from exc


class FitSession:
    """A fit that runs in a background thread and reports its progress.

    Parameters mirror :func:`esfit` (``data, model, p0, vary/lb/ub, options``); ``names``
    labels the parameters.  ``start()`` launches the fit (``esfit`` in a daemon thread,
    so the kernel stays responsive), ``stop()`` asks it to finish at the next evaluation
    (the result is then the best fit so far, ``interrupted=True``), ``result`` holds the
    :class:`FitResult` when done and ``history`` the (evaluations, best RMSD, elapsed)
    trace.  ``simulate(p)`` evaluates the model once, scaled to the data the way esfit
    does (least-squares amplitude and baseline).

    Bounds: with ``lb``/``ub`` (or neither, which defaults to ±10 % of ``p0``) the
    bounds are absolute and are kept across restarts and slider moves; with ``vary``
    they are ``p0 ± vary`` around the start of each run.  ``start(vary=...)`` with zero
    entries fixes those parameters for that run only (the panel's "fix" boxes), so a
    fixed parameter can be released again on the next run.

    ``vary`` is **not** clipped at anything physical here, because a plain parameter
    vector carries no information about what each entry means: ``vary`` larger than
    ``p0`` gives a negative lower bound, which for a linewidth or a weight lets the
    optimizer explore unphysical values.  Pass explicit ``lb``/``ub`` for those
    parameters (``lb=np.maximum(p0 - vary, floor)``), or use :func:`esfit`'s dict
    style, where the field names are known and non-negative fields are clipped at
    zero as EasySpin does.
    """

    def __init__(self, data, model: Callable, p0, vary=None, lb=None, ub=None,
                 options: Optional[FitOptions] = None, names: Optional[Sequence[str]] = None,
                 progress_interval: float = 0.5):
        self.data = np.asarray(data, dtype=float).ravel()
        self.model = model
        self.p0 = np.asarray(p0, dtype=float).ravel()
        n = self.p0.size
        self._explicit_bounds = vary is None
        if vary is not None:
            self.vary = np.asarray(vary, dtype=float).ravel()
            self.lb, self.ub = self.p0 - self.vary, self.p0 + self.vary
        else:
            self.lb = np.asarray(lb, dtype=float).ravel() if lb is not None else self.p0 - np.abs(self.p0) * 0.1 - 1e-3
            self.ub = np.asarray(ub, dtype=float).ravel() if ub is not None else self.p0 + np.abs(self.p0) * 0.1 + 1e-3
            self.vary = (self.ub - self.lb) / 2
        self.names = list(names) if names is not None else [f'p{i}' for i in range(n)]
        base = options if options is not None else FitOptions()
        self.options = replace(base, progress=self._on_progress, progress_interval=progress_interval,
                               stop_when=self._should_stop)
        self.history: list[tuple[int, float, float]] = []
        self.best_x: Optional[np.ndarray] = None
        self.best_f = float('inf')
        self.result: Optional[FitResult] = None
        self.error: Optional[BaseException] = None
        self.state = 'idle'
        self._stop = False
        self._thread: Optional[threading.Thread] = None
        self._dirty = False
        self._lock = threading.Lock()

    # -- esfit hooks (called from the fit thread) ---------------------------
    def _on_progress(self, info: dict):
        with self._lock:
            self.history.append((info['n_evaluations'], info['best_f'], info['elapsed_s']))
            if info['best_x'] is not None:
                # esfit reports the active (non-fixed) parameters; expand to the full vector
                self.best_x = self._to_full(np.asarray(info['best_x'], dtype=float))
                self.best_f = float(info['best_f'])
            self._dirty = True

    def _to_full(self, pvec_active: np.ndarray) -> np.ndarray:
        ph = getattr(self, '_ph', None)
        return np.asarray(ph.to_full(pvec_active) if ph is not None else pvec_active, dtype=float)

    def _should_stop(self, info: dict) -> bool:
        return self._stop

    # -- control --------------------------------------------------------------
    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, p0=None, vary=None, options: Optional[FitOptions] = None) -> None:
        if self.running:
            raise RuntimeError('a fit is already running; call stop() first')
        # Validate the run before touching any state, so a rejected start leaves the
        # session (p0, vary, options) exactly as it was.
        p0_run = self.p0.copy() if p0 is None else np.asarray(p0, dtype=float).ravel()
        vary_run = np.maximum(self.vary if vary is None else np.asarray(vary, dtype=float).ravel(), 0.0)
        if self._explicit_bounds:
            # Absolute bounds must survive slider changes and repeated fits.
            lb, ub = self.lb.copy(), self.ub.copy()
            if np.any(p0_run < lb) or np.any(p0_run > ub):
                raise ValueError('starting parameters must lie within lb/ub')
        else:
            lb, ub = p0_run - vary_run, p0_run + vary_run
        fixed = vary_run <= 0                     # per-run only: self.vary is not overwritten
        lb[fixed], ub[fixed] = p0_run[fixed], p0_run[fixed]
        self.p0 = p0_run
        if options is not None:
            self.options = replace(options, progress=self._on_progress,
                                   progress_interval=self.options.progress_interval, stop_when=self._should_stop)
        self._stop = False
        self.history = []
        self.result, self.error = None, None
        self.best_x, self.best_f = self.p0.copy(), float('inf')
        self.state = 'running'
        p0 = self.p0.copy()
        self._ph = ParameterHandler(p0, lb=lb, ub=ub)      # same active/fixed split esfit will use

        def _run():
            try:
                self.result = esfit(self.data, self.model, p0, lb=lb, ub=ub, options=self.options)
                with self._lock:
                    self.best_x = self._to_full(np.asarray(self.result.pfit, dtype=float))
                    self.best_f = float(self.result.rmsd)
                    self._dirty = True
                self.state = 'interrupted' if self.result.interrupted else 'done'
            except BaseException as e:      # noqa: BLE001 — surfaced through .error
                self.error = e
                self.state = 'error'

        self._thread = threading.Thread(target=_run, name='torchspin-fit', daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop = True

    def join(self, timeout: Optional[float] = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    # -- model evaluation for the plots -------------------------------------
    def simulate(self, p) -> np.ndarray:
        """Model output at ``p``, scaled to the data like esfit (lsq amplitude + baseline)."""
        sim = np.asarray(self.model(np.asarray(p, dtype=float)), dtype=float).ravel()
        if self.options.autoscale == 'lsq':
            scale, baseline = _fit_scale_baseline_lsq(
                sim, self.data, self.options.baseline,
                negative_scale=self.options.lsq_negative_scale)
            return float(scale) * sim + np.asarray(baseline, dtype=float)
        return sim

    def take_dirty(self) -> bool:
        with self._lock:
            d, self._dirty = self._dirty, False
        return d


class FitPanel:
    """ipywidgets panel around a :class:`FitSession`.

    ``panel.show()`` displays the panel (or put ``panel.widget`` as the last expression of a cell — not both);
    ``panel.result`` / ``panel.p_current`` give the fit result and the slider values.
    Sliders can be moved while idle (the starting simulation is redrawn); the "fix"
    boxes exclude parameters from the next fit; "Sliders → start" adopts the slider
    values as the new starting point.
    """

    def __init__(self, session: FitSession, figsize=(7.0, 6.5), refresh_s: float = 0.5, x=None, xlabel='B (mT)'):
        w = _require('ipywidgets', 'the parameter sliders and buttons')
        matplotlib = _require('matplotlib', 'the fit figure')
        plt = _require('matplotlib.pyplot', 'the fit figure')
        self.s = session
        self.refresh_s = refresh_s
        self.x = np.arange(self.s.data.size) if x is None else np.asarray(x, dtype=float)
        # --- parameter rows ---------------------------------------------------
        self.sliders, self.fixes = [], []
        rows = []
        for i, name in enumerate(self.s.names):
            lo, hi = float(self.s.lb[i]), float(self.s.ub[i])
            step = (hi - lo) / 1000.0 if hi > lo else 1e-3
            sl = w.FloatSlider(value=float(self.s.p0[i]), min=min(lo, hi), max=max(lo, hi), step=step, description=name,
                               readout_format='.5g', continuous_update=False, layout=w.Layout(width='340px'))
            fx = w.Checkbox(value=bool(self.s.vary[i] <= 0), description='fix', indent=False, layout=w.Layout(width='70px'))
            sl.observe(self._on_slider, names='value')
            self.sliders.append(sl); self.fixes.append(fx)
            rows.append(w.HBox([sl, fx]))
        o = self.s.options
        self.method = w.Dropdown(options=_METHODS, value=o.method if o.method in _METHODS else 'global', description='method')
        self.max_iter = w.IntText(value=int(o.max_iter), description='max_iter', layout=w.Layout(width='160px'))
        self.n_workers = w.Text(value=str(o.n_workers), description='n_workers', layout=w.Layout(width='160px'))
        self.max_time = w.FloatText(value=float(o.max_time or 0.0), description='max_time (s)', layout=w.Layout(width='180px'))
        self.b_start = w.Button(description='Start', button_style='success')
        self.b_stop = w.Button(description='Stop', button_style='danger', disabled=True)
        self.b_reset = w.Button(description='Reset sliders')
        self.b_adopt = w.Button(description='Sliders → start')
        self.b_start.on_click(lambda _: self.start()); self.b_stop.on_click(lambda _: self.stop())
        self.b_reset.on_click(lambda _: self.reset()); self.b_adopt.on_click(lambda _: self.adopt())
        self.status = w.HTML(value='<i>idle</i>')
        left = w.VBox(rows + [w.HBox([self.method, self.max_iter]), w.HBox([self.n_workers, self.max_time]),
                              w.HBox([self.b_start, self.b_stop, self.b_reset, self.b_adopt]), self.status])
        # --- figure -----------------------------------------------------------
        self._live = 'ipympl' in matplotlib.get_backend().lower() or 'widget' in matplotlib.get_backend().lower()
        with plt.ioff():
            self.fig, (self.ax_fit, self.ax_res, self.ax_trace) = plt.subplots(
                3, 1, figsize=figsize, gridspec_kw=dict(height_ratios=[3, 1, 1.4]))
        self.ax_fit.plot(self.x, self.s.data, color='0.6', lw=1, label='data')
        (self.l_start,) = self.ax_fit.plot(self.x, self.s.simulate(self.s.p0), 'k--', lw=0.9, label='start')
        (self.l_fit,) = self.ax_fit.plot(self.x, np.full_like(self.s.data, np.nan), 'r', lw=1.2, label='best fit')
        self.ax_fit.legend(fontsize=8); self.ax_fit.set_ylabel('intensity')
        (self.l_res,) = self.ax_res.plot(self.x, np.full_like(self.s.data, np.nan), 'b', lw=0.8)
        self.ax_res.axhline(0, color='k', lw=0.5); self.ax_res.set_ylabel('residual'); self.ax_res.set_xlabel(xlabel)
        (self.l_trace,) = self.ax_trace.plot([], [], 'o-', ms=2, lw=0.8)
        self.ax_trace.set_xlabel('evaluations'); self.ax_trace.set_ylabel('best RMSD'); self.ax_trace.set_yscale('log')
        self.fig.tight_layout()
        if self._live:
            try:
                self.fig.canvas.header_visible = False
            except Exception:
                pass
            right = self.fig.canvas
        else:
            self.img = w.Image(format='png')
            self._render_png()
            right = self.img
        self.widget = w.HBox([left, right])
        self._last_drawn_x = None
        self._poll_task = None
        self._start_polling()

    # -- state -------------------------------------------------------------------
    @property
    def p_current(self) -> np.ndarray:
        return np.array([sl.value for sl in self.sliders], dtype=float)

    @property
    def result(self) -> Optional[FitResult]:
        return self.s.result

    def show(self) -> None:
        """Display the panel (returns None so a trailing ``panel.show()`` is shown once)."""
        display = _require('IPython.display', 'rendering the panel in a notebook').display
        display(self.widget)

    # -- actions -----------------------------------------------------------------
    def start(self) -> None:
        if self.s.running:
            return
        vary = np.where([fx.value for fx in self.fixes], 0.0, self.s.vary)
        nw = self.n_workers.value.strip()
        n_workers = nw if nw == 'auto' else int(nw or 1)
        opts = replace(self.s.options, method=self.method.value, max_iter=int(self.max_iter.value),
                       n_workers=n_workers, max_time=float(self.max_time.value) or None)
        for sl in self.sliders:
            sl.disabled = True
        self.b_start.disabled, self.b_stop.disabled = True, False
        self.status.value = f'<b>running</b> {self.method.value} …'
        try:
            self.s.start(p0=self.p_current, vary=vary, options=opts)
        except Exception as e:                  # e.g. start outside lb/ub: hand the panel back
            for sl in self.sliders:
                sl.disabled = False
            self.b_start.disabled, self.b_stop.disabled = False, True
            self.status.value = f'<b>error</b>: {e!r}'
            raise
        self._start_polling()

    def stop(self) -> None:
        self.s.stop()
        self.status.value = '<b>stopping</b> at the next evaluation …'

    def reset(self) -> None:
        for sl, v in zip(self.sliders, self.s.p0):
            sl.value = float(v)

    def adopt(self) -> None:
        self.s.p0 = self.p_current
        self.l_start.set_ydata(self.s.simulate(self.s.p0)); self._draw()

    # -- drawing -----------------------------------------------------------------
    def _on_slider(self, change) -> None:
        if self.s.running:
            return
        self.l_start.set_ydata(self.s.simulate(self.p_current))
        self._draw()

    def refresh(self) -> None:
        """Redraw if the session reported progress (call manually without an event loop)."""
        s = self.s
        changed = s.take_dirty()
        if changed and s.best_x is not None and (self._last_drawn_x is None or not np.array_equal(s.best_x, self._last_drawn_x)):
            fit = s.simulate(s.best_x)
            self.l_fit.set_ydata(fit); self.l_res.set_ydata(s.data - fit)
            self._last_drawn_x = s.best_x.copy()
            self.ax_res.relim(); self.ax_res.autoscale_view()
        if changed and s.history:
            h = np.asarray(s.history); self.l_trace.set_data(h[:, 0], h[:, 1])
            self.ax_trace.relim(); self.ax_trace.autoscale_view()
        if s.history:
            n, f, t = s.history[-1]
            self.status.value = f'<b>{s.state}</b> · {n} evals · best RMSD {f:.4g} · {t:.1f} s'
        if not s.running and s.state in ('done', 'interrupted', 'error'):
            if self.b_stop.disabled is False:
                for sl, v in zip(self.sliders, s.best_x if s.best_x is not None else s.p0):
                    sl.disabled = False; sl.value = float(v)
                self.b_start.disabled, self.b_stop.disabled = False, True
                if s.error is not None:
                    self.status.value = f'<b>error</b>: {s.error!r}'
                elif s.result is not None:
                    self.status.value = (f'<b>{s.state}</b> · {s.result.n_evaluations} evals · RMSD {s.result.rmsd:.4g} · '
                                         f'{s.result.elapsed_s:.1f} s · {s.result.message}')
        if changed:
            self._draw()

    def _draw(self) -> None:
        if self._live:
            self.fig.canvas.draw_idle()
        else:
            self._render_png()

    def _render_png(self) -> None:
        buf = io.BytesIO()
        self.fig.savefig(buf, format='png', dpi=90)
        self.img.value = buf.getvalue()

    def _start_polling(self) -> None:
        """Poll the session at ``refresh_s`` on the kernel's event loop (Jupyter); without a
        running loop (scripts, tests) call :meth:`refresh` yourself."""
        try:
            import asyncio
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        if self._poll_task is not None and not self._poll_task.done():
            return

        async def _poll():
            while True:
                await asyncio.sleep(self.refresh_s)
                try:
                    self.refresh()
                except Exception:
                    pass
                if not self.s.running and self.s.state != 'idle':
                    self.refresh()
                    break
        self._poll_task = loop.create_task(_poll())
