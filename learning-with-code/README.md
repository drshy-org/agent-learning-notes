# Learning with Code

Code-grounded teardowns of real systems. Each post opens one model or system and walks its components in the order a single run hits them. For every component: the math, the actual code (with the exact file it lives in), how the choice differs from the common alternatives, and the papers behind it. Not from surveys or from memory — from the shipped source. All figures are drawn by hand.

## Posts

| # | post | one-liner |
|---|---|---|
| 1 | **[DiffusionGemma](diffusiongemma/diffusiongemma-deep-dive.md)** | A Gemma-4 MoE that writes by denoising whole 256-token blocks instead of going left-to-right. Every component traced to `google-deepmind/gemma` — RoPE, GQA, MoE, the diffusion loop, and the glue between them. |

---
*© 2026 Hongyu Shen. Original writing and figures, all rights reserved. Quoted code remains under its upstream license (e.g. Apache 2.0), with attribution.*
