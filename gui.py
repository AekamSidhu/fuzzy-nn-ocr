"""Interactive demo.  Run:  python gui.py

Pick a character (click a key or type it), or draw your own. Add noise, then
compare the two signal paths, threshold only and fuzzy enhancement, across
all five networks. "Animate recall" replays the Hopfield network settling,
neuron by neuron, with its energy falling.
"""
import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk

import numpy as np

import fuzzy
import models as M
from chars import CHARSET, GRID, N_PIXELS, from_bipolar, glyphs, to_bipolar
from noise import add_noise

# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------
SURFACE = "#E9EDF0"   # drafting-paper grey-blue
PANEL = "#FFFFFF"
INK = "#1B2430"       # navy ink: text and pixels
GRAPHITE = "#5B6672"  # secondary text
RULE = "#C9D1D8"      # hairlines
FUZZY = "#2A78D6"     # fuzzy path (same blue as the report figures)
THRESH = "#EB6834"    # threshold path (same orange as the report figures)
WRONG = "#D03B3B"

NOISE_NAMES = {"gaussian": "Gaussian", "saltpepper": "Salt and pepper", "faded": "Faded scan"}
LEVEL_RANGE = {"gaussian": (0.8, 0.4), "saltpepper": (0.6, 0.3), "faded": (0.35, 0.2)}  # (max, default)

STAGE_CELL, INPUT_CELL, RECALL_CELL = 4, 7, 3
ANIMATION_MS = 3000  # target length of the recall animation


def _rgb(hex_color: str) -> np.ndarray:
    return np.array([int(hex_color[i : i + 2], 16) for i in (1, 3, 5)], dtype=float)


_PAPER_RGB, _INK_RGB = _rgb(PANEL), _rgb(INK)


def shade(v: float) -> str:
    """Pixel value in [0, 1] (1 = ink) -> colour between paper and ink."""
    r, g, b = (_PAPER_RGB + (_INK_RGB - _PAPER_RGB) * float(np.clip(v, 0, 1))).round().astype(int)
    return f"#{r:02x}{g:02x}{b:02x}"


class Fonts:
    def __init__(self, root: tk.Tk):
        families = set(tkfont.families(root))
        base = "Bahnschrift" if "Bahnschrift" in families else "Segoe UI"
        bold = "Bahnschrift SemiBold" if "Bahnschrift SemiBold" in families else base
        light = "Bahnschrift Light" if "Bahnschrift Light" in families else base
        self.title = (bold, 19)
        self.lede = (light, 11)
        self.heading = (bold, 11)
        self.body = (base, 10)
        self.small = (base, 9)
        self.key = (bold, 11)
        self.answer = (bold, 24)


# ---------------------------------------------------------------------------
# Widgets
# ---------------------------------------------------------------------------
class PixelPanel(tk.Canvas):
    """A GRID x GRID image drawn as rectangles; optionally paintable."""

    def __init__(self, master, cell: int, on_edit=None):
        size = GRID * cell
        super().__init__(master, width=size, height=size, bg=PANEL, bd=0,
                         highlightthickness=1, highlightbackground=RULE)
        self.cell = cell
        self.rects = [self.create_rectangle(x * cell, y * cell, (x + 1) * cell, (y + 1) * cell, width=0, fill=PANEL)
                      for y in range(GRID) for x in range(GRID)]
        self.shown = np.full(N_PIXELS, -1.0)
        self.on_edit = on_edit
        self.dirty = False
        if on_edit is not None:
            self.image = np.zeros((GRID, GRID))
            self.configure(cursor="pencil")
            for btn, value in (("1", 1.0), ("3", 0.0)):
                self.bind(f"<Button-{btn}>", lambda e, v=value: self._paint(e, v))
                self.bind(f"<B{btn}-Motion>", lambda e, v=value: self._paint(e, v))
                self.bind(f"<ButtonRelease-{btn}>", lambda e: self._release())

    def show(self, img: np.ndarray) -> None:
        flat = np.round(np.asarray(img, dtype=float).reshape(-1), 2)
        for i in np.flatnonzero(flat != self.shown):
            self.itemconfigure(self.rects[i], fill=shade(flat[i]))
        self.shown = flat
        if self.on_edit is not None:
            self.image = np.asarray(img, dtype=float).copy()

    def _paint(self, event, value: float) -> None:
        x, y = event.x // self.cell, event.y // self.cell
        if 0 <= x < GRID and 0 <= y < GRID and self.image[y, x] != value:
            self.image[y, x] = value
            self.itemconfigure(self.rects[y * GRID + x], fill=shade(value))
            self.shown[y * GRID + x] = value
            self.dirty = True

    def _release(self) -> None:
        if self.dirty:
            self.dirty = False
            self.on_edit(self.image.copy())


class Arrow(tk.Canvas):
    """Horizontal arrow that stretches to its grid cell, with an optional caption."""

    def __init__(self, master, color: str, text: str = "", font=None, height: int = 24):
        super().__init__(master, height=height, width=36, bg=SURFACE, highlightthickness=0, bd=0)
        self.color, self.text, self.font = color, text, font
        self.bind("<Configure>", self._draw)

    def _draw(self, event) -> None:
        self.delete("all")
        w, h = event.width, event.height
        y = h - 6 if self.text else h // 2
        self.create_line(4, y, w - 4, y, fill=self.color, width=2, arrow="last", arrowshape=(9, 10, 4))
        if self.text:
            self.create_text(w // 2, y - 10, text=self.text, fill=self.color, font=self.font)


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------
class App:
    NETWORKS = (
        ("perceptron", "Perceptron", "Unit 1, all 36 characters"),
        ("hamming", "Hamming net + Maxnet", "Unit 3, all 36 characters"),
        ("hopfield", "Hopfield, projection rule", "Unit 2, all 36 characters"),
        ("hebbian", "Hopfield, Hebbian rule", "Unit 2, stores {mem} only"),
        ("bam", "BAM", "Unit 3, stores {mem} only"),
    )

    def __init__(self, root: tk.Tk):
        self.root = root
        self.f = Fonts(root)
        # Smaller pixel cells on short screens (laptops at high scaling, projectors).
        compact = root.winfo_screenheight() < 1000
        self.stage_px, self.input_px, self.recall_px = (3, 5, 2) if compact else (STAGE_CELL, INPUT_CELL, RECALL_CELL)
        root.title("Fuzzy-neural character recognizer")
        root.configure(bg=SURFACE)

        self.models = M.build()
        self.mem = self.models.memory_chars
        self.truth: str | None = "A"
        self.clean = glyphs("A")[0]
        self.kind = tk.StringVar(value="gaussian")
        self.level = tk.DoubleVar(value=LEVEL_RANGE["gaussian"][1])
        self.seed = 1
        self._pending = None
        self._anim = None

        page = self._build_scroll_page()
        self._build_header(page)
        body = tk.Frame(page, bg=SURFACE)
        body.pack(fill="both", expand=True, padx=24, pady=(0, 20))
        left = tk.Frame(body, bg=SURFACE)
        left.pack(side="left", fill="y", padx=(0, 28))
        right = tk.Frame(body, bg=SURFACE)
        right.pack(side="left", fill="both", expand=True)

        self._build_keypad(left)
        self._build_input(left)
        self._build_noise(left)
        self._build_stages(right)
        self._build_readout(right)

        root.bind("<Key>", self._on_key)
        root.protocol("WM_DELETE_WINDOW", self._close)
        self.select_char("A")
        self._fit_to_screen(page)

    def _close(self) -> None:
        self._stop_animation()
        self.root.destroy()

    def _build_scroll_page(self) -> tk.Frame:
        """A page inside a canvas, so the window can scroll when the screen is too short."""
        outer = tk.Frame(self.root, bg=SURFACE)
        outer.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(outer, bg=SURFACE, highlightthickness=0, bd=0)
        self.vbar = ttk.Scrollbar(outer, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.vbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        page = tk.Frame(self.canvas, bg=SURFACE)
        self.canvas.create_window((0, 0), window=page, anchor="nw")
        page.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        return page

    def _fit_to_screen(self, page: tk.Frame) -> None:
        """Size the window to its content; scroll only if the screen is too short for it."""
        self.root.update_idletasks()
        width, height = page.winfo_reqwidth(), page.winfo_reqheight()
        available = self.root.winfo_screenheight() - 110  # taskbar + title bar
        if height > available:
            self.vbar.pack(side="right", fill="y")
            self.root.bind_all("<MouseWheel>", lambda e: self.canvas.yview_scroll(-e.delta // 120, "units"))
            height = available
        self.canvas.configure(width=width, height=height)
        self.root.geometry("+0+0")

    # ----- layout ------------------------------------------------------------
    def _label(self, master, text, font, fg=INK, **kw):
        return tk.Label(master, text=text, font=font, fg=fg, bg=kw.pop("bg", SURFACE), **kw)

    def _button(self, master, text, command, **kw):
        return tk.Button(master, text=text, command=command, font=kw.pop("font", self.f.body),
                         bg=kw.pop("bg", PANEL), fg=kw.pop("fg", INK), activebackground=RULE, activeforeground=INK,
                         relief="flat", bd=0, padx=10, pady=5, cursor="hand2",
                         highlightthickness=2, highlightcolor=FUZZY, highlightbackground=SURFACE, **kw)

    def _build_header(self, master) -> None:
        head = tk.Frame(master, bg=SURFACE)
        head.pack(fill="x", padx=24, pady=(14, 10))
        self._label(head, "Fuzzy-neural character recognizer", self.f.title).pack(anchor="w")
        self._label(head, "Add noise to a character, then compare plain thresholding with fuzzy enhancement "
                          "across five neural networks.", self.f.lede, fg=GRAPHITE).pack(anchor="w")

    def _build_keypad(self, master) -> None:
        self._label(master, "Character", self.f.heading).pack(anchor="w")
        pad = tk.Frame(master, bg=SURFACE)
        pad.pack(anchor="w", pady=(6, 2))
        self.keys = {}
        for i, ch in enumerate(CHARSET):
            b = tk.Button(pad, text=ch, width=3, font=self.f.key, relief="flat", bd=0, bg=PANEL, fg=INK,
                          activebackground=RULE, cursor="hand2", highlightthickness=2, highlightcolor=FUZZY,
                          highlightbackground=SURFACE, underline=0 if ch in self.mem else -1,
                          command=lambda c=ch: self.select_char(c))
            b.grid(row=i // 9, column=i % 9, padx=1, pady=1, ipady=1)
            self.keys[ch] = b
        self._label(master, f"Underlined keys ({', '.join(self.mem)}) are also stored by the\nHebbian Hopfield net and BAM.",
                    self.f.small, fg=GRAPHITE, justify="left").pack(anchor="w", pady=(2, 14))

    def _build_input(self, master) -> None:
        row = tk.Frame(master, bg=SURFACE)
        row.pack(fill="x")
        self._label(row, "Input", self.f.heading).pack(side="left")
        self._button(row, "Clear", self.clear_drawing, font=self.f.small).pack(side="right")
        self.input_panel = PixelPanel(master, self.input_px, on_edit=self.drawn)
        self.input_panel.pack(anchor="w", pady=(6, 2))
        self._label(master, "Drag to draw, right-drag to erase.", self.f.small, fg=GRAPHITE).pack(anchor="w", pady=(0, 14))

    def _build_noise(self, master) -> None:
        self._label(master, "Noise", self.f.heading).pack(anchor="w")
        seg = tk.Frame(master, bg=RULE)
        seg.pack(anchor="w", pady=(6, 8))
        self.kind_buttons = {}
        for kind, name in NOISE_NAMES.items():
            b = tk.Radiobutton(seg, text=name, value=kind, variable=self.kind, indicatoron=False, font=self.f.body,
                               bg=PANEL, fg=INK, selectcolor=INK, activebackground=RULE, relief="flat", bd=0,
                               padx=10, pady=5, cursor="hand2", command=self._kind_changed)
            b.pack(side="left", padx=(0, 1))
            self.kind_buttons[kind] = b
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("Level.Horizontal.TScale", background=INK, troughcolor=PANEL, bordercolor=RULE,
                        lightcolor=INK, darkcolor=INK, sliderthickness=16)
        style.map("Level.Horizontal.TScale", background=[("active", FUZZY)])
        self.scale = ttk.Scale(master, variable=self.level, from_=0.0, to=LEVEL_RANGE["gaussian"][0],
                               orient="horizontal", length=290, style="Level.Horizontal.TScale",
                               command=lambda _v: self.schedule_refresh())
        self.scale.pack(anchor="w", pady=(2, 4))
        self.level_caption = self._label(master, "", self.f.small, fg=GRAPHITE)
        self.level_caption.pack(anchor="w", pady=(0, 8))
        self._button(master, "New noise sample", self.new_sample).pack(anchor="w")

    def _build_stages(self, master) -> None:
        self._label(master, "Signal paths", self.f.heading).pack(anchor="w")
        grid = tk.Frame(master, bg=SURFACE)
        grid.pack(anchor="w", pady=(4, 4))
        self._label(grid, "Fuzzy path: fuzzify, sharpen with INT, filter with 9 IF-THEN rules, then binarise",
                    self.f.small, fg=FUZZY).grid(row=0, column=2, columnspan=7, sticky="w", pady=(0, 4))
        self.stage = {}
        captions = [("noisy", "Noisy input"), ("fuzzified", "Fuzzified"), ("intensified", "After INT"),
                    ("filtered", "After rule filter"), ("fuzzy_bin", "Network input")]
        for i, (key, text) in enumerate(captions):
            col = i * 2
            self.stage[key] = PixelPanel(grid, self.stage_px)
            self.stage[key].grid(row=1, column=col)
            self._label(grid, text, self.f.small, fg=GRAPHITE).grid(row=2, column=col, pady=(2, 0))
            if i < len(captions) - 1:
                Arrow(grid, FUZZY).grid(row=1, column=col + 1, sticky="ew", padx=2)
        self._label(grid, "same noisy image", self.f.small, fg=GRAPHITE).grid(row=3, column=0, sticky="s")
        Arrow(grid, THRESH, "Threshold path: binarise at 0.5", self.f.small, height=40).grid(
            row=3, column=1, columnspan=7, sticky="ew", padx=2)
        self.stage["thr_bin"] = PixelPanel(grid, self.stage_px)
        self.stage["thr_bin"].grid(row=3, column=8, pady=(10, 0))
        self._label(grid, "Network input", self.f.small, fg=GRAPHITE).grid(row=4, column=8, pady=(2, 0))
        self.pixel_line = self._label(master, "", self.f.body, fg=INK)
        self.pixel_line.pack(anchor="w", pady=(2, 10))

    def _build_readout(self, master) -> None:
        table = tk.Frame(master, bg=RULE)
        table.pack(anchor="w", fill="x")
        heads = [("Network", INK), ("Threshold only", THRESH), ("Fuzzy enhancement", FUZZY)]
        for c, (text, color) in enumerate(heads):
            cell = tk.Frame(table, bg=PANEL, padx=12, pady=6)
            cell.grid(row=0, column=c, sticky="nsew", padx=(0 if c == 0 else 1, 0), pady=(0, 1))
            if c:
                tk.Frame(cell, bg=color, width=14, height=4).pack(side="left", padx=(0, 8))
            self._label(cell, text, self.f.heading, fg=color if c else INK, bg=PANEL).pack(side="left")
        table.columnconfigure(0, minsize=250)
        table.columnconfigure(1, minsize=290, weight=1)
        table.columnconfigure(2, minsize=290, weight=1)

        self.cells = {}
        for r, (key, name, note) in enumerate(self.NETWORKS, start=1):
            namecell = tk.Frame(table, bg=PANEL, padx=12, pady=6)
            namecell.grid(row=r, column=0, sticky="nsew", pady=(0, 1))
            self._label(namecell, name, self.f.heading, bg=PANEL).pack(anchor="w")
            self._label(namecell, note.format(mem=", ".join(self.mem)), self.f.small, fg=GRAPHITE, bg=PANEL).pack(anchor="w")
            if key == "hopfield":
                self.animate_btn = self._button(namecell, "Animate recall", self.animate, font=self.f.small, bg=SURFACE)
                self.animate_btn.pack(anchor="w", pady=(6, 0))
            for c, pipe in enumerate(M.PIPELINES, start=1):
                cell = tk.Frame(table, bg=PANEL, padx=12, pady=6)
                cell.grid(row=r, column=c, sticky="nsew", padx=(1, 0), pady=(0, 1))
                widgets = {}
                if key in ("hopfield", "hebbian"):
                    widgets["image"] = PixelPanel(cell, self.recall_px)
                    widgets["image"].pack(side="left", padx=(0, 12))
                text = tk.Frame(cell, bg=PANEL)
                text.pack(side="left", fill="x")
                widgets["answer"] = self._label(text, "", self.f.answer, bg=PANEL)
                widgets["answer"].pack(anchor="w")
                widgets["detail"] = self._label(text, "", self.f.small, fg=GRAPHITE, bg=PANEL, justify="left")
                widgets["detail"].pack(anchor="w")
                self.cells[(key, pipe)] = widgets

    # ----- interaction -------------------------------------------------------
    def _on_key(self, event) -> None:
        ch = (event.char or "").upper()
        if ch and ch in CHARSET and not isinstance(event.widget, tk.Scale):
            self.select_char(ch)

    def select_char(self, ch: str) -> None:
        self.truth = ch
        self.clean = glyphs(ch)[0]
        self._mark_key(ch)
        self.refresh()

    def drawn(self, image: np.ndarray) -> None:
        self.truth = None
        self.clean = image
        self._mark_key(None)
        self.refresh()

    def clear_drawing(self) -> None:
        self.drawn(np.zeros((GRID, GRID)))

    def new_sample(self) -> None:
        self.seed += 1
        self.refresh()

    def _kind_changed(self) -> None:
        top, default = LEVEL_RANGE[self.kind.get()]
        self.scale.configure(to=top)
        self.level.set(default)
        self.refresh()

    def _mark_key(self, active: str | None) -> None:
        for ch, b in self.keys.items():
            on = ch == active
            b.configure(bg=INK if on else PANEL, fg=PANEL if on else INK)

    def schedule_refresh(self) -> None:
        if self._pending is not None:
            self.root.after_cancel(self._pending)
        self._pending = self.root.after(80, self.refresh)

    # ----- computation -------------------------------------------------------
    def refresh(self) -> None:
        if self._pending is not None:
            self.root.after_cancel(self._pending)
            self._pending = None
        self._stop_animation()
        for kind, b in self.kind_buttons.items():
            b.configure(fg=PANEL if kind == self.kind.get() else INK)
        kind, level = self.kind.get(), round(float(self.level.get()), 2)
        self.level_caption.configure(text={
            "gaussian": f"Noise std σ = {level:.2f}",
            "saltpepper": f"{level:.0%} of pixels replaced by black or white",
            "faded": f"Grey paper, weak ink, noise std σ = {level:.2f}",
        }[kind])

        noisy = add_noise(self.clean, kind, level, np.random.default_rng(self.seed))
        stages = fuzzy.enhance_stages(noisy)
        x = {"threshold": to_bipolar(noisy), "fuzzy": to_bipolar(stages["filtered"])}
        self.inputs = x

        self.input_panel.show(self.clean)
        self.stage["noisy"].show(noisy)
        for key in ("fuzzified", "intensified", "filtered"):
            self.stage[key].show(stages[key])
        self.stage["fuzzy_bin"].show(from_bipolar(x["fuzzy"]))
        self.stage["thr_bin"].show(from_bipolar(x["threshold"]))

        ink = self.clean >= 0.5
        wrong = {p: int(((from_bipolar(x[p]) > 0) != ink).sum()) for p in M.PIPELINES}
        self.pixel_line.configure(text=f"Pixels wrong at the network input: threshold {wrong['threshold']}, "
                                       f"fuzzy {wrong['fuzzy']} (of {N_PIXELS})")

        for pipe in M.PIPELINES:
            self._fill_row(pipe, x[pipe])

    def _verdict(self, predicted: str | None, stored: str) -> tuple[str, str]:
        """Mark (symbol, colour) for a prediction, given which characters the network knows."""
        if self.truth is None or self.truth not in stored:
            return "", INK
        return ("✓", INK) if predicted == self.truth else ("✗", WRONG)

    def _set(self, key: str, pipe: str, answer: str | None, detail: str, stored: str) -> None:
        mark, color = self._verdict(answer, stored)
        w = self.cells[(key, pipe)]
        w["answer"].configure(text=f"{answer if answer else 'none'} {mark}".strip(), fg=color)
        w["detail"].configure(text=detail)

    def _fill_row(self, pipe: str, x: np.ndarray) -> None:
        m = self.models
        net = m.perceptrons[pipe].net_input(x)[0]
        top2 = np.argsort(net)[::-1][:2]
        self._set("perceptron", pipe, CHARSET[top2[0]],
                  f"net input {net[top2[0]]:.0f}, next best {CHARSET[top2[1]]} at {net[top2[1]]:.0f}", CHARSET)

        agree = m.hamming.agreements(x)[0]
        winner, iters = m.hamming.predict(x, return_iters=True)
        self._set("hamming", pipe, CHARSET[winner[0]],
                  f"{agree[winner[0]]:.0f} of {N_PIXELS} pixels match,\nMaxnet settled in {iters[0]} steps", CHARSET)

        for key, hop, stored in (("hopfield", m.hopfield, CHARSET), ("hebbian", m.hopfield_hebb, self.mem)):
            state = hop.recall(x, seed=0)
            self.cells[(key, pipe)]["image"].show(from_bipolar(state))
            idx = hop.match(state)[0]
            answer = stored[idx] if idx >= 0 else None
            detail = f"energy {hop.energy(state)[0]:.0f}\n" + ("recalled a stored pattern" if idx >= 0 else "spurious state")
            self._set(key, pipe, answer, detail, stored)

        _, y, steps = m.bam.recall(x)
        code_idx = int((y @ m.bam.codes.T).argmax(axis=1)[0])
        match = int((y[0] == m.bam.codes[code_idx]).sum())
        self._set("bam", pipe, self.mem[code_idx],
                  f"code matches {match} of {m.bam.codes.shape[1]} bits,\nstable after {steps} passes", self.mem)

    # ----- Hopfield animation ---------------------------------------------------
    def animate(self) -> None:
        self._stop_animation()
        frames = {}
        for pipe in M.PIPELINES:
            _, hist = self.models.hopfield.recall(self.inputs[pipe], seed=0, record=True)
            changed = [hist[0]] + [h for prev, h in zip(hist, hist[1:]) if not np.array_equal(prev[0], h[0])]
            frames[pipe] = changed
        n = max(len(f) for f in frames.values())
        per_tick = max(1, int(np.ceil(n / (ANIMATION_MS / 30))))
        self.animate_btn.configure(state="disabled", text="Recalling...")
        self._tick(frames, 0, per_tick)

    def _tick(self, frames: dict, step: int, per_tick: int) -> None:
        done = True
        for pipe, seq in frames.items():
            i = min(step, len(seq) - 1)
            state, energy = seq[i]
            w = self.cells[("hopfield", pipe)]
            w["image"].show(from_bipolar(state))
            w["answer"].configure(text="…", fg=GRAPHITE)
            w["detail"].configure(text=f"energy {energy:.0f}\nflip {i} of {len(seq) - 1}")
            done &= step >= len(seq) - 1
        if done:
            self._anim = None
            self.animate_btn.configure(state="normal", text="Animate recall")
            for pipe in M.PIPELINES:
                self._fill_row(pipe, self.inputs[pipe])
            return
        self._anim = self.root.after(30, self._tick, frames, step + per_tick, per_tick)

    def _stop_animation(self) -> None:
        if self._anim is not None:
            self.root.after_cancel(self._anim)
            self._anim = None
            self.animate_btn.configure(state="normal", text="Animate recall")


def main() -> None:
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
