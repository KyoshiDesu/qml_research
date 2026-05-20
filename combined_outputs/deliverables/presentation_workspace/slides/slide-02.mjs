import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, "..", "..");
const summary = JSON.parse(fs.readFileSync(path.join(root, "data", "summary.json"), "utf8"));
const accuracyChart = path.join(root, "assets", "accuracy_comparison.png");

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

export async function slide02(presentation, ctx) {
  const slide = presentation.slides.add();
  ctx.addShape(slide, { left: 0, top: 0, width: 1280, height: 720, fill: "#F6F1E8", line: ctx.line("#F6F1E8", 0) });
  ctx.addShape(slide, { left: 64, top: 52, width: 8, height: 28, fill: "#1F5EFF", line: ctx.line("#1F5EFF", 0), name: "kicker-marker" });
  addText(ctx, slide, "Accuracy", 84, 50, 150, 30, { fontSize: 13, bold: true, color: "#5A6476", name: "kicker-label" });
  addText(ctx, slide, "Accuracy leadership stays with the classical baseline in three of four datasets.", 64, 92, 820, 64, { fontSize: 28, bold: true, serif: true, face: "Georgia" });

  await ctx.addImage(slide, { path: accuracyChart, left: 64, top: 176, width: 760, height: 466, fit: "contain", name: "accuracy-chart" });

  addPanel(ctx, slide, 866, 176, 350, 466, "#FFFDF9", "right-panel");
  const entries = [...summary.datasets].sort((a, b) => b.accuracy_gap - a.accuracy_gap);
  entries.forEach((entry, idx) => {
    addPanel(ctx, slide, 892, 204 + idx * 104, 298, 92, idx === 0 ? "#FFF3EB" : "#FFFDF9");
    addText(ctx, slide, entry.display_name, 914, 216 + idx * 106, 150, 24, { fontSize: 16, bold: true });
    addText(
      ctx,
      slide,
      `SVM ${Math.round(entry.svm_accuracy * 1000) / 10}% | QSVM ${Math.round(entry.qsvm_accuracy * 1000) / 10}%\n${Math.abs(entry.accuracy_gap) < 0.0001 ? "Tie" : `Gap: ${Math.round(entry.accuracy_gap * 1000) / 10} pts`} | ${entry.rows} rows | ${entry.features} features`,
      914,
      246 + idx * 104,
      240,
      40,
      { fontSize: 13, color: "#5A6476", bold: false }
    );
  });

  addText(ctx, slide, "Interpretation: the empirical case for QSVM is weakest on Wine and Breast Cancer, where the accuracy gap is meaningful and the classical model remains cleaner.", 64, 654, 1080, 28, { fontSize: 14, color: "#5A6476" });
  addText(ctx, slide, "2", 1190, 650, 30, 24, { fontSize: 12, color: "#5A6476", align: "right" });
  return slide;
}
