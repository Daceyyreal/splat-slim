"""splat-slim command-line interface.

Examples:
    splat-slim info scene.ply
    splat-slim prune scene.ply a.ply --percentile 5
    splat-slim reduce-sh a.ply b.ply --degree 2
    splat-slim run scene.ply slim.ply --degree 3 --quant mixed
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from . import __version__, io
from .stages import outliers, prune, quantize, sh

app = typer.Typer(add_completion=False, help="Shrink 3D Gaussian Splatting models, training-free.")
console = Console()


def _save(out: str, fields, quant: str):
    """Save either a plain (fp16/none) fields dict or a quantized PLY."""
    if quant in ("fp16", "int8", "mixed"):
        structured, comments = quantize.quantize(fields, quant)
        io.save_quantized(out, structured, comments)
    else:
        io.save_splat(out, fields)


@app.command()
def info(ply: str):
    """Print Gaussian count, fields, and file size."""
    fields = io.load_splat(ply)
    n = len(next(iter(fields.values())))
    table = Table(title=ply)
    table.add_column("metric")
    table.add_column("value")
    table.add_row("gaussians", f"{n:,}")
    table.add_row("fields", str(len(fields)))
    table.add_row("size (MB)", f"{io.filesize_mb(ply):.1f}")
    console.print(table)


@app.command(name="prune")
def prune_cmd(inp: str, out: str, percentile: float = 5.0):
    """Stage 1: adaptive opacity pruning."""
    io.save_splat(out, prune.prune_opacity(io.load_splat(inp), percentile))
    console.print(f"[green]pruned[/] -> {out} ({io.filesize_mb(out):.1f} MB)")


@app.command()
def clean(inp: str, out: str, scale_cap: float = 1.0):
    """Stage 2: spatial/scale outlier removal."""
    io.save_splat(out, outliers.remove_outliers(io.load_splat(inp), scale_cap=scale_cap))
    console.print(f"[green]cleaned[/] -> {out} ({io.filesize_mb(out):.1f} MB)")


@app.command(name="reduce-sh")
def reduce_sh(inp: str, out: str, degree: int = 2):
    """Stage 3: SH degree reduction (1/2/3)."""
    io.save_splat(out, sh.reduce_sh(io.load_splat(inp), degree))
    console.print(f"[green]sh->{degree}[/] -> {out} ({io.filesize_mb(out):.1f} MB)")


@app.command(name="quantize")
def quantize_cmd(inp: str, out: str, mode: str = "mixed"):
    """Stage 4: quantization (fp16 / int8 / mixed)."""
    _save(out, io.load_splat(inp), mode)
    console.print(f"[green]quant:{mode}[/] -> {out} ({io.filesize_mb(out):.1f} MB)")


@app.command()
def run(inp: str, out: str, percentile: float = 5.0, scale_cap: float = 1.0,
        degree: int = 3, quant: str = "mixed"):
    """Run the full pipeline: prune -> clean -> reduce-sh -> quantize."""
    f = io.load_splat(inp)
    f = prune.prune_opacity(f, percentile)
    f = outliers.remove_outliers(f, scale_cap=scale_cap)
    f = sh.reduce_sh(f, degree)
    _save(out, f, quant)
    console.print(f"[bold green]done[/] -> {out} ({io.filesize_mb(out):.1f} MB)")


@app.command()
def version():
    """Print version."""
    console.print(__version__)


if __name__ == "__main__":
    app()
