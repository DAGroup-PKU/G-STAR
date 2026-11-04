// Prints a WebVTT transcript generated from src/captions.ts.
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

const src = readFileSync(fileURLToPath(new URL('../src/captions.ts', import.meta.url)), 'utf8');
const re = /start:\s*([\d.]+),\s*end:\s*([\d.]+),\s*text:\s*'((?:[^'\\]|\\.)*)'/g;
const ts = s => {
  const ms = Math.round(s * 1000);
  const h = String(Math.floor(ms / 3600000)).padStart(2, '0');
  const m = String(Math.floor(ms / 60000) % 60).padStart(2, '0');
  const sec = String(Math.floor(ms / 1000) % 60).padStart(2, '0');
  return `${h}:${m}:${sec}.${String(ms % 1000).padStart(3, '0')}`;
};
let out = 'WEBVTT\n';
let i = 0;
for (const [, a, b, text] of src.matchAll(re)) out += `\n${++i}\n${ts(+a)} --> ${ts(+b)}\n${text.replace(/\\'/g, "'")}\n`;
process.stdout.write(out);
