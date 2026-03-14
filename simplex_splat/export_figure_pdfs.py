#!/usr/bin/env python3
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from PIL import Image
from pypdf import PdfReader, PdfWriter


ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = ROOT / "report" / "figures"
BUILD_DIR = FIGURES_DIR / "_pdf_build"
OUT_DIR = FIGURES_DIR / "pdf"
PNG_DIR = FIGURES_DIR / "png"


WRAPPER_TEMPLATE = r"""
\documentclass{article}
\usepackage[paperwidth=8in,paperheight=6in,margin=0.35in]{geometry}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{pgfplots}
\usepackage{tikz}
\usetikzlibrary{shapes,arrows.meta,positioning,fit,backgrounds,calc,decorations.pathreplacing}
\usepgfplotslibrary{fillbetween}
\pgfplotsset{compat=1.18}
\usepackage{xcolor}

\definecolor{tealaccent}{HTML}{20808D}
\definecolor{terracotta}{HTML}{A84B2F}
\definecolor{darkteal}{HTML}{1B474D}
\definecolor{lightcyan}{HTML}{BCE2E7}
\definecolor{mauvecol}{HTML}{944454}
\definecolor{goldcol}{HTML}{FFC553}
\definecolor{safezone}{HTML}{2E8B57}
\definecolor{dangerzone}{HTML}{C0392B}

\begin{document}
\pagestyle{empty}
    hispagestyle{empty}
\centering
\input{%INPUT%}
\end{document}
""".strip()


def run_cmd(command: list[str], cwd: Path) -> None:
    completed = subprocess.run(command, cwd=cwd, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed ({' '.join(command)}):\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )


def trim_png_whitespace(png_path: Path, margin_px: int = 24) -> tuple[int, int, int, int] | None:
    with Image.open(png_path) as img:
        rgb = img.convert("RGB")
        # Pixels that are not near-white count as figure content.
        mask = Image.eval(rgb, lambda v: 255 if v < 250 else 0).convert("L")
        bbox = mask.getbbox()
        if not bbox:
            return None

        left, top, right, bottom = bbox
        left = max(0, left - margin_px)
        top = max(0, top - margin_px)
        right = min(rgb.width, right + margin_px)
        bottom = min(rgb.height, bottom + margin_px)

        cropped = rgb.crop((left, top, right, bottom))
        cropped.save(png_path)
        return (left, top, right, bottom)


def crop_pdf_to_pixel_bbox(pdf_path: Path, bbox_px: tuple[int, int, int, int], image_size: tuple[int, int]) -> None:
    reader = PdfReader(str(pdf_path))
    if not reader.pages:
        return

    page = reader.pages[0]
    width_pt = float(page.mediabox.width)
    height_pt = float(page.mediabox.height)
    img_w, img_h = image_size
    left, top, right, bottom = bbox_px

    x0 = left / img_w * width_pt
    x1 = right / img_w * width_pt
    y0 = (img_h - bottom) / img_h * height_pt
    y1 = (img_h - top) / img_h * height_pt

    # Safety margin in PDF points to avoid edge antialias clipping.
    margin_pt = 4.0
    x0 = max(0.0, x0 - margin_pt)
    y0 = max(0.0, y0 - margin_pt)
    x1 = min(width_pt, x1 + margin_pt)
    y1 = min(height_pt, y1 + margin_pt)

    page.mediabox.lower_left = (x0, y0)
    page.mediabox.upper_right = (x1, y1)
    page.cropbox.lower_left = (x0, y0)
    page.cropbox.upper_right = (x1, y1)

    writer = PdfWriter()
    writer.add_page(page)
    with pdf_path.open("wb") as f:
        writer.write(f)


def convert_pdf_to_png(pdf_path: Path, png_path: Path) -> bool:
    pdftocairo = shutil.which("pdftocairo")
    if pdftocairo:
        # -singlefile avoids suffixes like -1 and writes exactly the requested basename.
        run_cmd([pdftocairo, "-png", "-singlefile", "-r", "300", str(pdf_path), str(png_path.with_suffix(""))], ROOT)
        return True

    pdftoppm = shutil.which("pdftoppm")
    if pdftoppm:
        run_cmd([pdftoppm, "-png", "-r", "300", str(pdf_path), str(png_path.with_suffix(""))], ROOT)
        generated = png_path.with_suffix("").with_name(png_path.stem + "-1").with_suffix(".png")
        if generated.exists():
            generated.replace(png_path)
        return png_path.exists()

    qlmanage = shutil.which("qlmanage")
    if qlmanage:
        # Quick Look can rasterize PDF at high resolution on macOS.
        run_cmd([qlmanage, "-t", "-s", "3000", "-o", str(png_path.parent), str(pdf_path)], ROOT)
        quicklook_output = png_path.parent / f"{pdf_path.name}.png"
        if quicklook_output.exists():
            quicklook_output.replace(png_path)
            return True

    # Last fallback: sips is widely available but often low resolution.
    sips = shutil.which("sips")
    if not sips:
        return False
    run_cmd([sips, "-s", "format", "png", str(pdf_path), "--out", str(png_path)], ROOT)
    return True


def tighten_outputs(pdf_path: Path, png_path: Path) -> None:
    with Image.open(png_path) as img:
        image_size = (img.width, img.height)

    bbox = trim_png_whitespace(png_path)
    if not bbox:
        return

    crop_pdf_to_pixel_bbox(pdf_path, bbox, image_size)


def main() -> None:
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)

    figure_files = sorted(
        path for path in FIGURES_DIR.glob("*.tex")
        if path.name != "main.tex"
    )

    failed: list[str] = []
    converted_png = 0

    for figure_path in figure_files:
        wrapper_name = figure_path.stem + "_standalone.tex"
        wrapper_path = BUILD_DIR / wrapper_name
        input_path = f"../{figure_path.name}"
        wrapper_path.write_text(WRAPPER_TEMPLATE.replace("%INPUT%", input_path))

        try:
            run_cmd(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", wrapper_name], BUILD_DIR)
            pdf_src = BUILD_DIR / (figure_path.stem + "_standalone.pdf")
            pdf_dst = OUT_DIR / f"{figure_path.stem}.pdf"
            pdf_dst.write_bytes(pdf_src.read_bytes())
            print(f"Wrote PDF: {pdf_dst}")

            png_dst = PNG_DIR / f"{figure_path.stem}.png"
            if convert_pdf_to_png(pdf_dst, png_dst):
                tighten_outputs(pdf_dst, png_dst)
                converted_png += 1
                print(f"Wrote PNG: {png_dst}")
        except Exception as exc:  # noqa: BLE001
            failed.append(f"{figure_path.name}: {exc}")
            print(f"FAILED: {figure_path.name}")

    if failed:
        print("\nFailures:")
        for msg in failed:
            print(f"- {msg}")
        raise SystemExit(1)

    print(f"\nDone: exported {len(figure_files)} PDFs and {converted_png} PNGs.")


if __name__ == "__main__":
    main()