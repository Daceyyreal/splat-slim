"""splat-slim command-line interface.

Examples:
    splat-slim info scene.ply
    splat-slim prune scene.ply a.ply --percentile 5
    splat-slim reduce-sh a.ply b.ply --degree 2
    splat-slim run scene.ply slim.ply --degree 3 --quant mixed
    splat-slim run scene.ply slim.ply --degree 3 --quant int8-subgroup   # + slim.meta.npz
    splat-slim dequantize slim.ply restored.ply
"""

from __future__ import annotations

from enum import Enum

import typer
from rich.console import Console
from rich.table import Table

from . import __version__, io
from .stages import outliers, prune, quantize, sh

app = typer.Typer(add_completion=False, help="Shrink 3D Gaussian Splatting models, training-free.")
console = Console()


class Quant(str, Enum):
    """Storage modes for the quantization stage."""

    none = "none"                      # plain float32 PLY
    fp16 = "fp16"                      # every field FP16
    int8 = "int8"                      # every field INT8, ONE range per field (collapses geometry)
    mixed = "mixed"                    # FP16 geometry + INT8 appearance
    int8_subgroup = "int8-subgroup"    # every field INT8, G local ranges per field (+ sidecar)


_GROUPS_HELP = "Sub-group count G for --quant int8-subgroup (paper: ~1000). Ignored by other modes."
_CAP_HELP = (
    "Hard cap on Gaussian size in LINEAR units; stored log-scales must be <= log(scale_cap). "
    "The default 1.0 is the paper's cap (log-scale 0). Use 'inf' to disable."
)


def _save(out: str, fields, quant: Quant, groups: int) -> str:
    """Write ``fields`` in the requested storage mode; return a human-readable size summary."""
    if quant is Quant.int8_subgroup:
        structured, meta = quantize.int8_subgroup(fields, groups)
        side = io.save_subgroup(out, structured, meta)
        return (f"{io.filesize_mb(out):.1f} MB + {io.filesize_mb(side):.2f} MB sidecar "
                f"({side.name}, {meta.n_groups} groups)")
    if quant in (Quant.fp16, Quant.int8, Quant.mixed):
        structured, comments = quantize.quantize(fields, quant.value)
        io.save_quantized(out, structured, comments)
    else:
        io.save_splat(out, fields)
    return f"{io.filesize_mb(out):.1f} MB"


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
def clean(inp: str, out: str, scale_cap: float = typer.Option(1.0, help=_CAP_HELP)):
    """Stage 2: spatial/scale outlier removal."""
    io.save_splat(out, outliers.remove_outliers(io.load_splat(inp), scale_cap=scale_cap))
    console.print(f"[green]cleaned[/] -> {out} ({io.filesize_mb(out):.1f} MB)")


@app.command(name="reduce-sh")
def reduce_sh(inp: str, out: str, degree: int = 2):
    """Stage 3: SH degree reduction (1/2/3)."""
    io.save_splat(out, sh.reduce_sh(io.load_splat(inp), degree))
    console.print(f"[green]sh->{degree}[/] -> {out} ({io.filesize_mb(out):.1f} MB)")


@app.command(name="quantize")
def quantize_cmd(
    inp: str,
    out: str,
    mode: Quant = Quant.mixed,
    groups: int = typer.Option(quantize.DEFAULT_GROUPS, min=1, help=_GROUPS_HELP),
):
    """Stage 4: quantization (fp16 / int8 / mixed / int8-subgroup)."""
    summary = _save(out, io.load_splat(inp), mode, groups)
    console.print(f"[green]quant:{mode.value}[/] -> {out} ({summary})")


@app.command()
def run(
    inp: str,
    out: str,
    percentile: float = 5.0,
    scale_cap: float = typer.Option(1.0, help=_CAP_HELP),
    degree: int = 3,
    quant: Quant = Quant.mixed,
    groups: int = typer.Option(quantize.DEFAULT_GROUPS, min=1, help=_GROUPS_HELP),
):
    """Run the full pipeline: prune -> clean -> reduce-sh -> quantize."""
    f = io.load_splat(inp)
    f = prune.prune_opacity(f, percentile)
    f = outliers.remove_outliers(f, scale_cap=scale_cap)
    f = sh.reduce_sh(f, degree)
    summary = _save(out, f, quant, groups)
    console.print(f"[bold green]done[/] -> {out} ({summary})")


@app.command()
def dequantize(inp: str, out: str):
    """Restore a float32 PLY from a sub-group INT8 file (needs <inp>.meta.npz beside it).

    The output is in the stored (Morton) order, not the original Gaussian order.
    """
    try:
        fields = io.load_subgroup(inp)
    except (FileNotFoundError, ValueError) as err:
        console.print(f"[red]error:[/] {err}")
        raise typer.Exit(1) from err
    io.save_splat(out, fields)
    console.print(f"[green]dequantized[/] -> {out} ({io.filesize_mb(out):.1f} MB)")


@app.command()
def version():
    """Print version."""
    console.print(__version__)


if __name__ == "__main__":
    app()
