import fs from "node:fs/promises";
import path from "node:path";
import crypto from "node:crypto";

const [sourcePath, outputPath, mappingPath] = process.argv.slice(2);

if (!sourcePath || !outputPath || !mappingPath) {
  throw new Error("Usage: node prepare_annotator_b_blind.mjs <source.jsonl> <blind.json> <mapping.json>");
}

const sourceText = await fs.readFile(sourcePath, "utf8");
const source = sourceText
  .split(/\r?\n/)
  .filter((line) => line.trim())
  .map((line) => JSON.parse(line));

if (source.length !== 700) {
  throw new Error(`Expected exactly 700 source records; received ${source.length}.`);
}

const shuffleSeed = "RQ3-main700-blind-2026-08-26-v1";
const shuffled = source
  .map((record, sourceIndex) => ({
    record,
    sourceIndex,
    sortKey: crypto.createHash("sha256").update(`${shuffleSeed}:${sourceIndex + 1}`).digest("hex"),
  }))
  .sort((a, b) => a.sortKey.localeCompare(b.sortKey));

const mapping = [];
const blind = shuffled.map(({ record, sourceIndex }, blindIndex) => {
  for (const field of ["question", "context", "factoid"]) {
    if (typeof record[field] !== "string" || !record[field].trim()) {
      throw new Error(`Record ${sourceIndex + 1} has an invalid ${field}.`);
    }
  }

  const annotationId = `RQ3-${String(blindIndex + 1).padStart(4, "0")}`;
  mapping.push({ annotation_id: annotationId, source_row: sourceIndex + 1 });

  return {
    annotation_id: annotationId,
    question: record.question,
    context: record.context,
    factoid: record.factoid,
  };
});

const outputText = `${JSON.stringify(blind, null, 2)}\n`;
await fs.mkdir(path.dirname(outputPath), { recursive: true });
await fs.writeFile(outputPath, outputText, "utf8");
await fs.mkdir(path.dirname(mappingPath), { recursive: true });
await fs.writeFile(mappingPath, `${JSON.stringify({ shuffleSeed, mapping }, null, 2)}\n`, "utf8");

const sourceSha256 = crypto.createHash("sha256").update(sourceText).digest("hex");
const blindSha256 = crypto.createHash("sha256").update(outputText).digest("hex");
process.stdout.write(JSON.stringify({ count: blind.length, sourceSha256, blindSha256 }, null, 2));
