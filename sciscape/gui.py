"""Minimal Tkinter GUI for SciScape landscape pipeline.

Launch with ``sciscape gui`` or ``python -m sciscape.gui``.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
except ImportError:
    raise ImportError(
        "tkinter is required for the GUI. Install it with:\n"
        "  Ubuntu/Debian: sudo apt install python3-tk\n"
        "  Fedora: sudo dnf install python3-tkinter\n"
        "  macOS: brew install python-tk\n"
        "  conda: conda install tk"
    )

log = logging.getLogger(__name__)


class _LogHandler(logging.Handler):
    """Route log messages to a Tkinter Text widget (thread-safe)."""

    def __init__(self, text_widget: tk.Text, app: "SciScapeApp"):
        super().__init__()
        self._text = text_widget
        self._app = app

    def emit(self, record: logging.LogRecord):
        msg = self.format(record) + "\n"
        self._text.after(0, self._append, msg)
        # Parse progress hints from log messages
        self._text.after(0, self._app._parse_progress, record.getMessage())

    def _append(self, msg: str):
        self._text.configure(state="normal")
        self._text.insert("end", msg)
        self._text.see("end")
        self._text.configure(state="disabled")


class SciScapeApp:
    """Main application window."""

    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("SciScape")
        root.geometry("680x560")
        root.resizable(True, True)

        self._running = False
        self._thread: threading.Thread | None = None

        self._build_ui()

    # ── UI construction ───────────────────────────────────────

    def _build_ui(self):
        root = self.root
        pad = dict(padx=8, pady=4)

        # ── File selection frame ──
        file_frame = ttk.LabelFrame(root, text="Input / Output", padding=8)
        file_frame.pack(fill="x", **pad)

        self.edge_var = tk.StringVar()
        self.abs_var = tk.StringVar()
        self.out_var = tk.StringVar(value="landscape_output")

        for row, (label, var, ftypes) in enumerate([
            ("Edge file:", self.edge_var, [("Parquet", "*.parquet"), ("All", "*.*")]),
            ("Abstract file:", self.abs_var, [("Parquet", "*.parquet"), ("All", "*.*")]),
            ("Output dir:", self.out_var, None),
        ]):
            ttk.Label(file_frame, text=label, width=14, anchor="e").grid(
                row=row, column=0, sticky="e", padx=(0, 4), pady=2,
            )
            entry = ttk.Entry(file_frame, textvariable=var, width=48)
            entry.grid(row=row, column=1, sticky="ew", pady=2)
            if ftypes is not None:
                btn = ttk.Button(
                    file_frame, text="Browse",
                    command=lambda v=var, ft=ftypes: self._browse_file(v, ft),
                )
            else:
                btn = ttk.Button(
                    file_frame, text="Browse",
                    command=lambda v=var: self._browse_dir(v),
                )
            btn.grid(row=row, column=2, padx=(4, 0), pady=2)

        file_frame.columnconfigure(1, weight=1)

        # ── Parameters frame ──
        param_frame = ttk.LabelFrame(root, text="Parameters", padding=8)
        param_frame.pack(fill="x", **pad)

        self.min_docs_var = tk.StringVar(value="1000")
        self.gamma_pre_var = tk.StringVar(value="auto")
        self.n_nodes_var = tk.StringVar(value="100000")
        self.seed_var = tk.StringVar(value="42")

        params = [
            ("Min docs/cluster:", self.min_docs_var, 0, 0),
            ("γ block:", self.gamma_pre_var, 0, 2),
            ("Target nodes:", self.n_nodes_var, 1, 0),
            ("Seed:", self.seed_var, 1, 2),
        ]
        for label_text, var, r, c in params:
            ttk.Label(param_frame, text=label_text, anchor="e").grid(
                row=r, column=c, sticky="e", padx=(8, 4), pady=2,
            )
            ttk.Entry(param_frame, textvariable=var, width=14).grid(
                row=r, column=c + 1, sticky="w", pady=2,
            )

        # ── Buttons ──
        btn_frame = ttk.Frame(root)
        btn_frame.pack(fill="x", **pad)

        self.run_btn = ttk.Button(btn_frame, text="Run", command=self._on_run)
        self.run_btn.pack(side="left", padx=4)

        self.open_btn = ttk.Button(
            btn_frame, text="Open Report", command=self._on_open, state="disabled",
        )
        self.open_btn.pack(side="left", padx=4)

        self.viewer_btn = ttk.Button(
            btn_frame, text="Export Viewer", command=self._on_export_viewer,
        )
        self.viewer_btn.pack(side="right", padx=4)

        # ── Progress ──
        self.progress = ttk.Progressbar(root, mode="indeterminate")
        self.progress.pack(fill="x", **pad)

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(root, textvariable=self.status_var, anchor="w").pack(
            fill="x", padx=8,
        )

        # ── Log output ──
        log_frame = ttk.LabelFrame(root, text="Log", padding=4)
        log_frame.pack(fill="both", expand=True, **pad)

        self.log_text = tk.Text(
            log_frame, height=12, wrap="word", font=("Consolas", 9),
            state="disabled", background="#1a1a2e", foreground="#e0e0e0",
        )
        scrollbar = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.log_text.pack(fill="both", expand=True)

        # Install log handler
        self._log_handler = _LogHandler(self.log_text, self)
        self._log_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S"))
        logging.getLogger("sciscape").addHandler(self._log_handler)
        logging.getLogger("sciscape").setLevel(logging.INFO)

    # ── File dialogs ──────────────────────────────────────────

    def _browse_file(self, var: tk.StringVar, filetypes: list):
        path = filedialog.askopenfilename(filetypes=filetypes)
        if path:
            var.set(path)

    def _browse_dir(self, var: tk.StringVar):
        path = filedialog.askdirectory()
        if path:
            var.set(path)

    # ── Progress parsing ──────────────────────────────────────

    def _parse_progress(self, msg: str):
        """Extract status hints from log messages."""
        if "Block init" in msg:
            self.status_var.set("Block initialization...")
        elif "Contracted" in msg:
            self.status_var.set("Graph contraction...")
        elif "Searching optimal" in msg:
            self.status_var.set("Searching resolution parameter...")
        elif "Cascade" in msg:
            self.status_var.set("Cascade search...")
        elif "Refinement round" in msg:
            self.status_var.set("Refining clusters...")
        elif "Building CPM dendrogram" in msg:
            self.status_var.set("Building hierarchy...")
        elif "keyword extraction" in msg.lower():
            self.status_var.set("Extracting keywords...")
        elif "Landscape complete" in msg or "nano membership saved" in msg:
            self.status_var.set("Clustering complete")

    # ── Run pipeline ──────────────────────────────────────────

    def _validate(self) -> dict | None:
        edge = self.edge_var.get().strip()
        abstract = self.abs_var.get().strip()
        output = self.out_var.get().strip()

        if not edge:
            messagebox.showerror("Error", "Select an edge file.")
            return None
        if not Path(edge).exists():
            messagebox.showerror("Error", f"Edge file not found:\n{edge}")
            return None
        if not abstract:
            messagebox.showerror("Error", "Select an abstract file.")
            return None
        if not Path(abstract).exists():
            messagebox.showerror("Error", f"Abstract file not found:\n{abstract}")
            return None
        if not output:
            messagebox.showerror("Error", "Select an output directory.")
            return None

        # Parse gamma_pre
        gb = self.gamma_pre_var.get().strip().lower()
        if gb == "none":
            gamma_pre = None
        elif gb == "auto":
            gamma_pre = "auto"
        else:
            try:
                gamma_pre = float(gb)
            except ValueError:
                messagebox.showerror("Error", f"Invalid γ block: {gb}")
                return None

        try:
            min_docs = int(self.min_docs_var.get())
            n_nodes = int(self.n_nodes_var.get())
            seed = int(self.seed_var.get())
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid parameter: {e}")
            return None

        return dict(
            edge_path=edge,
            abstract_path=abstract,
            output_dir=output,
            gamma_pre=gamma_pre,
            min_docs=min_docs,
            n_nodes=n_nodes,
            seed=seed,
        )

    def _on_run(self):
        if self._running:
            return

        params = self._validate()
        if params is None:
            return

        self._running = True
        self.run_btn.configure(state="disabled")
        self.open_btn.configure(state="disabled")
        self.progress.start(10)
        self.status_var.set("Starting pipeline...")

        # Clear log
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

        self._thread = threading.Thread(
            target=self._run_pipeline, args=(params,), daemon=True,
        )
        self._thread.start()

    def _run_pipeline(self, params: dict):
        try:
            from .landscape import LandscapeConfig, run_landscape

            cfg = LandscapeConfig(
                n_target_nodes=params["n_nodes"],
                seed=params["seed"],
                min_docs_per_cluster=params["min_docs"],
                gamma_pre=params["gamma_pre"],
            )
            self._result = run_landscape(
                params["edge_path"],
                params["abstract_path"],
                params["output_dir"],
                config=cfg,
            )
            self.root.after(0, self._on_done, None)
        except Exception as e:
            self.root.after(0, self._on_done, e)

    def _on_done(self, error: Exception | None):
        self._running = False
        self.progress.stop()
        self.run_btn.configure(state="normal")

        if error:
            self.status_var.set(f"Error: {error}")
            messagebox.showerror("Pipeline Error", str(error))
        else:
            self.status_var.set("Done!")
            self.open_btn.configure(state="normal")
            messagebox.showinfo(
                "Complete",
                f"Results saved to:\n{self._result['report_dir']}",
            )

    # ── Open report ───────────────────────────────────────────

    def _on_open(self):
        if not hasattr(self, "_result"):
            return
        import webbrowser
        report_dir = Path(self._result["report_dir"])
        report_html = report_dir / "report.html"
        if report_html.exists():
            webbrowser.open(f"file://{report_html.resolve()}")
        else:
            messagebox.showwarning("Not found", f"Report not found:\n{report_html}")

    # ── Export viewer ─────────────────────────────────────────

    def _on_export_viewer(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".html",
            filetypes=[("HTML", "*.html")],
            initialfile="viewer.html",
        )
        if not path:
            return
        from .keyword_extraction.visualization import export_viewer
        abs_path = export_viewer(output_path=path)
        messagebox.showinfo("Viewer Exported", f"Saved: {abs_path}\n\nDeploy to GitHub Pages or open locally.")


def launch():
    """Entry point for ``sciscape gui``."""
    root = tk.Tk()
    SciScapeApp(root)
    root.mainloop()


if __name__ == "__main__":
    launch()
