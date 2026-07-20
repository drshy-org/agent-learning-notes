# Agent Learning Notes

*My notes from learning how AI agents actually work.*

I'm working through modern AI agents — and the systems they're built on — and writing down what I learn along the way. The notes are records for myself, first: writing them down keeps me honest. Keeping them public is for sharing the learning — anyone reading along, comparing notes, or correcting me is welcome. They cover whatever the learning touches — model internals, algorithms, systems, papers.

Notes are organized into series, one folder per series:

| series | what it is |
|---|---|
| **[learning-with-code/](learning-with-code/)** | Code-grounded teardowns — one real model or system at a time, walked component by component from the shipped source. |
| **[learning-with-maps/](learning-with-maps/)** | Question-driven field maps — one research question walked through an entire area: lineages, formalisms, open verdicts. |

## Latest

- **[Learning with Maps: Nobody Hands Out the Roles](learning-with-maps/division-of-labor/division-of-labor.md)** — where does division of labor come from when no one assigns the roles? MARL's five-year answer, the LLM-agent world's three-year rediscovery, and the four signals that would settle it.
- **[Learning with Code: Nemotron 3](learning-with-code/nemotron3/nemotron3-deep-dive.md)** — NVIDIA's 550B-A55B flagship that keeps attention in only 12 of its 108 blocks. Mamba-2, latent-space experts, and the draft model riding in the checkpoint, traced through `huggingface/transformers` and `vllm`.
- **[Learning with Code: DiffusionGemma](learning-with-code/diffusiongemma/diffusiongemma-deep-dive.md)** — a Gemma-4 MoE that writes by denoising whole 256-token blocks instead of going left-to-right. Every component traced to `google-deepmind/gemma`.

Also on Substack: [*Agent Learning Notes*](https://hongyushen.substack.com) · Site: [drshy.xyz](https://www.drshy.xyz)

Corrections are always welcome — I'd rather fix an error than leave it wrong. (Issues are open.)

---
*© 2026 Hongyu Shen. Original writing and figures, all rights reserved. Quoted code remains under its upstream license (e.g. Apache 2.0), with attribution.*
