import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, "..", "..");
const summary = JSON.parse(fs.readFileSync(path.join(root, "data", "summary.json"), "utf8"));

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
    valign: options.valign ?? "top",
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

export async function slide01(presentation, ctx) {
  const slide = presentation.slides.add();
  ctx.addShape(slide, { left: 0, top: 0, width: 1280, height: 720, fill: "#F6F1E8", line: ctx.line("#F6F1E8", 0) });
  ctx.addShape(slide, { left: 64, top: 56, width: 8, height: 24, fill: "#172033", line: ctx.line("#172033", 0), name: "kicker-marker" });
  addText(ctx, slide, "QML Research", 86, 53, 220, 30, { fontSize: 13, bold: true, color: "#5A6476", name: "kicker-label", valign: "middle" });
  addText(ctx, slide, "Combined outputs synthesis", 86, 84, 320, 24, { fontSize: 13, color: "#5A6476" });

  addText(ctx, slide, "Classical SVM matched or beat QSVM on every dataset in the experiment pack.", 64, 164, 680, 146, {
    fontSize: 34,
    bold: true,
    serif: true,
    face: "Georgia",
  });
  addText(ctx, slide, "The strongest signal in the logs is not a breakthrough from QSVM, but a persistent compute tradeoff: much longer search time for equal or weaker test accuracy.", 64, 330, 600, 86, {
    fontSize: 18,
    color: "#5A6476",
  });

  addPanel(ctx, slide, 760, 154, 450, 420, "#FFFDF9", "hero-panel");
  const findings = summary.portfolio_findings;
  findings.forEach((finding, idx) => {
    addPanel(ctx, slide, 790, 186 + idx * 88, 390, 70, idx === 0 ? "#F2F7FF" : "#FFFCF6");
    addText(ctx, slide, `0${idx + 1}`, 812, 208 + idx * 88, 40, 28, { fontSize: 20, bold: true, color: idx % 2 === 0 ? "#1F5EFF" : "#F06C3B" });
    addText(ctx, slide, finding, 860, 200 + idx * 88, 300, 46, { fontSize: 15, bold: idx === 0, color: "#172033" });
  });

  addText(ctx, slide, "Source base: logs, metadata, tables, and plots in `combined_outputs`.", 64, 650, 620, 22, { fontSize: 12, color: "#5A6476" });
  addText(ctx, slide, "1", 1190, 650, 30, 24, { fontSize: 12, color: "#5A6476", align: "right" });
  return slide;
}
