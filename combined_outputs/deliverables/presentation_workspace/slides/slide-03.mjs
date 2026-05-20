import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, "..", "..");
const summary = JSON.parse(fs.readFileSync(path.join(root, "data", "summary.json"), "utf8"));
const runtimeChart = path.join(root, "assets", "runtime_comparison.png");

function addText(ctx, slide, text, left, top, width, height, options = {}) {
  return ctx.addText(slide, {
    text,
    left,
    top,
    width,
    height,
    fontSize: options.fontSize ?? 24,
    bold: options.bold ?? false,
    color: options.color ?? "#172033",
    face: options.face ?? (options.serif ? "Georgia" : "Aptos"),
    align: options.align ?? "left",
    name: options.name,
  });
}

function addPanel(ctx, slide, left, top, width, height, fill = "#FFFDF9", name) {
  return ctx.addShape(slide, {
    left,
    top,
    width,
    height,
    geometry: "roundRect",
    fill,
    line: ctx.line("#D8CBB6", 1.5),
    name,
  });
}

export async function slide03(presentation, ctx) {
  const slide = presentation.slides.add();
  ctx.addShape(slide, { left: 0, top: 0, width: 1280, height: 720, fill: "#F6F1E8", line: ctx.line("#F6F1E8", 0) });
  ctx.addShape(slide, { left: 64, top: 52, width: 8, height: 28, fill: "#F06C3B", line: ctx.line("#F06C3B", 0), name: "kicker-marker" });
  addText(ctx, slide, "Runtime", 84, 50, 150, 30, { fontSize: 13, bold: true, color: "#5A6476", name: "kicker-label" });
  addText(ctx, slide, "The runtime story is decisive: QSVM search consumes the budget long before it creates an accuracy advantage.", 64, 92, 840, 64, { fontSize: 28, bold: true, serif: true, face: "Georgia" });

  await ctx.addImage(slide, { path: runtimeChart, left: 64, top: 180, width: 760, height: 470, fit: "contain", name: "runtime-chart" });

  addPanel(ctx, slide, 860, 178, 360, 472, "#FFFDF9", "runtime-panel");
  const richest = [...summary.datasets].sort((a, b) => b.total_qsvm_seconds - a.total_qsvm_seconds);
  richest.slice(0, 3).forEach((entry, idx) => {
    addPanel(ctx, slide, 888, 206 + idx * 122, 304, 102, idx === 0 ? "#FFF3EB" : "#FFFDF9");
    addText(ctx, slide, entry.display_name, 910, 220 + idx * 122, 160, 24, { fontSize: 16, bold: true });
    addText(
      ctx,
      slide,
      `QSVM build: ${(entry.total_qsvm_seconds / 60).toFixed(1)} min | ${Math.round(entry.total_qsvm_seconds / Math.max(entry.total_classical_seconds, 0.001))}x slower\nClassical build: ${entry.total_classical_seconds.toFixed(1)} s\n${entry.gpu_available ? (entry.gpu_used ? "GPU visible, but still not the main accelerator." : "GPU visible, yet effectively idle.") : "No GPU detected in the recorded runtime summary."}`,
      910,
      248 + idx * 122,
      240,
      52,
      { fontSize: 12, color: "#5A6476" }
    );
  });

  addText(ctx, slide, "Implication: any future QSVM iteration should begin with search-space reduction and pipeline profiling, not broader dataset expansion.", 64, 654, 1080, 28, { fontSize: 14, color: "#5A6476" });
  addText(ctx, slide, "3", 1180, 650, 30, 24, { fontSize: 12, color: "#5A6476", align: "right" });
  return slide;
}
