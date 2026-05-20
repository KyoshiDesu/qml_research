import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, "..", "..");
const summary = JSON.parse(fs.readFileSync(path.join(root, "data", "summary.json"), "utf8"));
const logFindings = path.join(root, "assets", "log_findings.png");
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

export async function slide04(presentation, ctx) {
  const slide = presentation.slides.add();
  ctx.addShape(slide, { left: 0, top: 0, width: 1280, height: 720, fill: "#F6F1E8", line: ctx.line("#F6F1E8", 0) });
  ctx.addShape(slide, { left: 64, top: 52, width: 8, height: 28, fill: "#0E766E", line: ctx.line("#0E766E", 0), name: "kicker-marker" });
  addText(ctx, slide, "Logs and diagnostics", 84, 50, 220, 30, { fontSize: 13, bold: true, color: "#5A6476", name: "kicker-label" });
  addText(ctx, slide, "The logs show a pipeline that eventually completes, but not one that is yet stable enough to call mature.", 64, 92, 830, 64, { fontSize: 28, bold: true, serif: true, face: "Georgia" });

  await ctx.addImage(slide, { path: logFindings, left: 64, top: 188, width: 640, height: 380, fit: "contain", name: "log-findings-image" });
  addPanel(ctx, slide, 742, 188, 474, 380, "#FFFDF9", "notes-panel");
  const iris = summary.datasets.find((entry) => entry.key === "iris");
  const heart = summary.datasets.find((entry) => entry.key === "heart");
  const noteLines = [
    "The first Iris attempt failed before the classical search even began because the SVM grid passed a scalar `C` instead of a list.",
    "All four datasets then logged repeated QSVM Optuna trial failures with the same `NoneType` error signature.",
    "Even so, the final reruns completed and exported the expected reports, tables, and decision-boundary plots.",
    `Heart Disease is the only case with recorded GPU visibility, yet its runtime summary still flagged several phases as mostly idle on GPU.`,
  ];
  noteLines.forEach((line, idx) => {
    addText(ctx, slide, line, 768, 214 + idx * 78, 410, 60, { fontSize: 15, color: "#172033", bold: idx === 0 });
  });

  addPanel(ctx, slide, 64, 590, 1152, 84, "#FFFDF9", "closing-strip");
  addText(ctx, slide, "Operational takeaway", 88, 606, 200, 22, { fontSize: 14, bold: true, color: "#0E766E" });
  addText(ctx, slide, "Fix the QSVM trial-failure path and standardize the compute environment before expanding the experiment suite.", 296, 604, 820, 36, { fontSize: 14, color: "#172033" });
  addText(ctx, slide, "4", 1180, 684, 30, 20, { fontSize: 12, color: "#5A6476", align: "right" });
  return slide;
}
