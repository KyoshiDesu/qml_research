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

export async function slide06(presentation, ctx) {
  const slide = presentation.slides.add();
  ctx.addShape(slide, { left: 0, top: 0, width: 1280, height: 720, fill: "#172033", line: ctx.line("#172033", 0) });
  ctx.addShape(slide, { left: 64, top: 52, width: 8, height: 28, fill: "#F6F1E8", line: ctx.line("#F6F1E8", 0), name: "kicker-marker" });
  addText(ctx, slide, "Recommendations", 84, 50, 180, 30, { fontSize: 13, bold: true, color: "#D8E0EE", name: "kicker-label" });
  addText(ctx, slide, "What to do next", 64, 96, 420, 56, { fontSize: 32, bold: true, serif: true, face: "Georgia", color: "#FFFFFF" });

  const cards = [
    ["1", "Keep classical SVM as the benchmark and likely default.", "It wins or ties everywhere in this experiment pack."],
    ["2", "Stabilize the QSVM search path before scaling.", "The repeated Optuna `NoneType` trial failures are a reliability warning, not just noise."],
    ["3", "Make runtime studies apples-to-apples.", "Heart Disease ran on stronger hardware than the other datasets, so future comparisons should normalize the environment."],
    ["4", "Only revisit QSVM with a narrower question.", "Good candidates are feature-map ablations, smaller search spaces, and profiling-focused runs rather than broad reruns."],
  ];

  cards.forEach((card, idx) => {
    const x = idx % 2 === 0 ? 64 : 654;
    const y = idx < 2 ? 188 : 424;
    addPanel(ctx, slide, x, y, 560, 176, idx === 0 ? "#243451" : "#1E2C45");
    addText(ctx, slide, card[0], x + 24, y + 24, 36, 28, { fontSize: 22, bold: true, color: "#F6F1E8" });
    addText(ctx, slide, card[1], x + 78, y + 24, 420, 28, { fontSize: 18, bold: true, color: "#FFFFFF" });
    addText(ctx, slide, card[2], x + 78, y + 64, 438, 62, { fontSize: 14, color: "#D8E0EE" });
  });

  addText(ctx, slide, "Bottom line: the current evidence supports classical SVM as the more efficient and more reliable choice in this experiment set.", 64, 646, 980, 28, { fontSize: 15, color: "#D8E0EE" });
  addText(ctx, slide, "6", 1190, 650, 30, 24, { fontSize: 12, color: "#D8E0EE", align: "right" });
  return slide;
}
