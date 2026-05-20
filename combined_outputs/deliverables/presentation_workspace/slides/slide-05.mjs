import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, "..", "..");
const summary = JSON.parse(fs.readFileSync(path.join(root, "data", "summary.json"), "utf8"));
const breastPlot = path.join(root, "..", "outputs_breast", "plots", "svm_decision_boundary_pca2d.png");
const heartPlot = path.join(root, "..", "outputs_heart", "plots", "qsvm_decision_boundary_pca2d.png");

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

export async function slide05(presentation, ctx) {
  const slide = presentation.slides.add();
  ctx.addShape(slide, { left: 0, top: 0, width: 1280, height: 720, fill: "#F6F1E8", line: ctx.line("#F6F1E8", 0) });
  ctx.addShape(slide, { left: 64, top: 52, width: 8, height: 28, fill: "#172033", line: ctx.line("#172033", 0), name: "kicker-marker" });
  addText(ctx, slide, "Dataset drill-down", 84, 50, 200, 30, { fontSize: 13, bold: true, color: "#5A6476", name: "kicker-label" });
  addText(ctx, slide, "The practical decision is clearest on Breast Cancer and Heart Disease: classical SVM stays ahead without the quantum tax.", 64, 92, 840, 64, { fontSize: 28, bold: true, serif: true, face: "Georgia" });

  addPanel(ctx, slide, 64, 182, 554, 446, "#FFFDF9", "left-visual-panel");
  addPanel(ctx, slide, 662, 182, 554, 446, "#FFFDF9", "right-visual-panel");
  await ctx.addImage(slide, { path: breastPlot, left: 86, top: 208, width: 510, height: 278, fit: "contain", name: "breast-plot" });
  await ctx.addImage(slide, { path: heartPlot, left: 684, top: 208, width: 510, height: 278, fit: "contain", name: "heart-plot" });

  const breast = summary.datasets.find((entry) => entry.key === "breast");
  const heart = summary.datasets.find((entry) => entry.key === "heart");

  addText(ctx, slide, "Breast Cancer", 92, 498, 180, 22, { fontSize: 16, bold: true });
  addText(ctx, slide, `Accuracy: ${(breast.svm_accuracy * 100).toFixed(1)}% vs ${(breast.qsvm_accuracy * 100).toFixed(1)}%`, 92, 526, 320, 22, { fontSize: 13, color: "#172033" });
  addText(ctx, slide, breast.confusion_note, 92, 552, 450, 44, { fontSize: 12, color: "#5A6476" });
  addText(ctx, slide, `${(breast.total_qsvm_seconds / 60).toFixed(1)} min QSVM build time`, 430, 526, 156, 22, { fontSize: 12, color: "#F06C3B", bold: true, align: "right" });

  addText(ctx, slide, "Heart Disease", 690, 498, 180, 22, { fontSize: 16, bold: true });
  addText(ctx, slide, `Accuracy: ${(heart.svm_accuracy * 100).toFixed(1)}% vs ${(heart.qsvm_accuracy * 100).toFixed(1)}%`, 690, 526, 320, 22, { fontSize: 13, color: "#172033" });
  addText(ctx, slide, heart.confusion_note, 690, 552, 450, 44, { fontSize: 12, color: "#5A6476" });
  addText(ctx, slide, `${(heart.total_qsvm_seconds / 60).toFixed(1)} min QSVM build time`, 1028, 526, 156, 22, { fontSize: 12, color: "#F06C3B", bold: true, align: "right" });

  addText(ctx, slide, "These plots are illustrative rather than decisive on their own, but they pair well with the tabular results: QSVM does not convert its extra search time into better generalization on the harder datasets.", 64, 650, 1080, 28, { fontSize: 14, color: "#5A6476" });
  addText(ctx, slide, "5", 1190, 650, 30, 24, { fontSize: 12, color: "#5A6476", align: "right" });
  return slide;
}
