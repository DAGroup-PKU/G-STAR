// Single source of truth for on-screen captions and the exported .vtt transcript (times in seconds).
export const CAPTIONS: { start: number; end: number; text: string }[] = [
  { start: 0.3, end: 3.8, text: 'Diffusion UMMs keep three copies of every image, and learn by denoising.' },
  { start: 4.0, end: 7.3, text: 'A discrete model needs just one copy, and every token is a prediction target.' },
  { start: 7.7, end: 10.8, text: 'The catch: it paints one token at a time, so high resolution means long, slow sequences.' },
  { start: 11.2, end: 14.7, text: 'Compress harder and the sequence gets short, but each token must carry far more detail.' },
  { start: 15.3, end: 17.8, text: 'Meet GSTAR: group-scaled token capacity.' },
  { start: 18.3, end: 21.4, text: 'Vision encoders like Qwen-ViT already merge 2×2 sub-tokens along the channel.' },
  { start: 21.6, end: 24.5, text: 'Quantize each one before merging, and every position holds a group of codes.' },
  { start: 24.9, end: 27.7, text: 'Double the compression, quadruple the group: the bit budget stays the same.' },
  { start: 27.9, end: 30.8, text: 'Packing 2×2 turns a 32× tokenizer into an effective 64× one, losslessly.' },
  { start: 31.3, end: 35.1, text: 'A causal head is slow; a parallel head is fast, but its codes can collide.' },
  { start: 35.3, end: 39.2, text: 'Our semi-autoregressive head: one parallel pass, then a one-layer causal sweep.' },
  { start: 39.8, end: 44.2, text: 'With system optimizations, a 1024 × 1024 image takes just 2.5 s on one H200,' },
  { start: 44.7, end: 49.2, text: 'up to 21× faster than prior discrete AR models, with no loss in quality.' },
  { start: 49.8, end: 53.2, text: 'It edits images too, scoring 7.45 on GEdit-Bench.' },
  { start: 53.8, end: 57.6, text: 'GSTAR. Short sequences, rich tokens, high-quality images.' },
];
