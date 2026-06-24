# Hardware Upgrade — Dedicated Voice Assistant Server

## Status & deployment plan (June 2026)

**Deployment plan.** The M1 16GB is the owner's personal MacBook (travels Paris ⇄ Charras), not a fixed instance — each fixed site needs its own dedicated server.
- **Charras first** — server on site for the July–August summer stay; real-world test bench. Charras also needs a fresh Home Assistant install (Velux skylights + Legrand lighting + Magiline pool via community HACS plugin). Idea: co-host HA (Docker/VM) + voice server on the same box → `HA_URL=localhost`, one box, no extra Pi.
- **Paris in September** (la rentrée) — second server, *only if* Charras validates.

**Availability / pricing reality check (June 2026).** The recommended Zotac Magnus EN275060TC became effectively unbuyable:
- Barebone: cheap listings sold out; only Fnac left at **~2,900€** (real price was ~1,600€).
- Windows SKU (`EN275060TC-W`): **~2,760€** — no relief, both SKUs spiked together.
- DDR5 SO-DIMM surged (AI-driven DRAM shortage): a single 32GB DDR5-5600 SO-DIMM is now **~338€** — the ~80€ estimate further down is stale.
- Zotac build at current prices ≈ **~3,380€** (2,900 + 338 RAM + 140 SSD). Not worth it.

**Pivot — prebuilt tower with RTX 5060 Ti 16GB.** In stock at Materiel.net / LDLC, shipped <24h, ~**1,700–1,800€** all-in (RAM + SSD + Win11 bundled → dodges the DDR5 surge; wipe to Ubuntu Server). Loses the 2.65L form factor — irrelevant for a closet server. Concrete configs (June 2026, ⚠️ verify the GPU is the **16GB** variant, not 8GB):
- **Materiel.net "PC Gamer Hellfest Valley"** — i5-14400F, 32GB DDR5, 1TB NVMe — 1,999€ (~1,700€ with code `GAME`), pre-assembled & tested. ← recommended.
- **Materiel.net "PC Gamer Viper 16G"** — i5-14600KF, 32GB DDR5, 1TB NVMe — 2,049€ (~1,800€ with code `GAME`), 5-yr warranty, 24h.
- **DIY alternative** ≈ 1,150–1,250€ (desktop RTX 5060 Ti 16GB from ~329€ + ITX/mATX parts) — saves ~500€ but assembly + multi-part order.

**Porting caveat for an x86+CUDA server.** STT (Parakeet) and TTS (Kokoro) currently use MLX (Apple-only). On the new box they must run on CUDA/CPU and share the 16GB VRAM with the LLM (~2GB for STT+TTS → ~14GB left, OK for Gemma 4 26B A4B Q4). Separate porting effort.

> The sections below predate this update — they keep the original Zotac recommendation and the (now stale) ~1,700€ total for reference.

## Why

The M1 16GB runs the pipeline but the LLM (Qwen 3 4B Q4_K_M) is limited:
- ~21 tok/s, ~1.7s warm latency
- Inconsistent tool calling → lots of Python compensation (room groups, text tool call parser, generic entity names, hardcoded TTS responses)
- 8GB max for the LLM (STT + TTS take the rest)

A dedicated GPU with 16GB VRAM would allow a 14B+ model that handles all of this natively.

## Recommendation: Zotac Magnus EN275060TC (Barebone)

- **GPU**: RTX 5060 Ti 16GB GDDR7 (desktop, not laptop)
- **CPU**: Intel Core Ultra 7 255HX (20 cores, 5.2 GHz turbo)
- **Form factor**: 2.65L (21 × 20.3 × 6.2 cm), smaller than a PS5
- **Connectivity**: WiFi 7, 2x 2.5GbE, Thunderbolt 4, HDMI + DP
- **Product page**: https://www.zotac.com/us/product/mini_pcs/magnus-en275060tc-barebone

### Required additions (barebone = no RAM, SSD, OS)

| Component | Spec | Estimated price |
|---|---|---|
| Barebone | Zotac Magnus EN275060TC | ~1,558€ ([idealo.fr](https://www.idealo.fr/prix/208347852/zotac-zbox-magnus-en275060tc.html)) |
| RAM | 1x SO-DIMM DDR5-5600 32GB | ~80€ |
| SSD | M.2 NVMe 512GB or 1TB | ~40-60€ |
| OS | Ubuntu Server 24.04 | free |
| **Total** | | **~1,700€** |

### What it can run

| Model | Quantization | VRAM | Estimated tok/s |
|---|---|---|---|
| Qwen 3 8B | Q8_0 (near native) | ~9GB | ~50 tok/s |
| Qwen 3 14B | Q4_K_M | ~8.5GB | ~35 tok/s |
| Qwen 3 14B | Q6_K | ~11GB | ~30 tok/s |
| Qwen 3 32B | Q3_K_M | ~14GB | ~15 tok/s |
| **Gemma 4 26B A4B** | **Q4** | **~10-14GB** | **~50-80 tok/s** |
| Gemma 4 31B | Q4_K_M | ~20-24GB | ❌ won't fit |

New sweet spot: **Gemma 4 26B A4B** — MoE architecture (25.2B total, 3.8B active, 128 experts). Only 3.8B params active per token = fast inference with 26B-level knowledge. Native function calling baked into the model. Apache 2.0 license. Released April 2026.

Previous sweet spot: ~~Qwen 3 14B Q6_K~~ — still a strong fallback, but Gemma 4 26B A4B is likely faster (2-3x) with better tool calling at similar VRAM usage. Both should be benchmarked once hardware is available.

### Python code that could be simplified/removed

With a reliable 14B+ model for tool calling:
- `ROOM_GROUPS` and `resolve_all_entities()` → model would make multiple tool calls
- `_parse_text_tool_call()` → reliable structured tool calls
- `_GENERIC_NAMES` → model would understand "les volets" = all covers
- Hardcoded TTS responses → model generates natural responses
- `/no_think` hack → no longer needed with a model that handles latency

## Cheaper alternative: Minisforum G1 Pro

- **GPU**: RTX 5060 8GB GDDR7
- **CPU**: AMD Ryzen 9 8945HX (16 cores)
- **Price**: ~1,440€ with 32GB RAM + 1TB SSD (ready to use)
- **Limitation**: 8GB VRAM = max Qwen 3 8B Q8_0, no 14B. Gemma 4 26B A4B might also fit (people run it on 12GB), but tight
- **Link**: https://minisforumpc.eu/products/minisforum-atomman-g1-pro-minipc

## Comparative benchmark (warm latency, tool calling)

| Setup | Model | Warm latency | tok/s |
|---|---|---|---|
| M1 16GB (current) | Qwen 3 4B Q4_K_M | ~1.7s | ~21 |
| M1 16GB | Qwen 3 8B Q4_K_M | ~3.1s | ~12 |
| M1 16GB | Gemma 4 E4B (to test) | ? | ? |
| RTX 5060 Ti 16GB (estimated) | Qwen 3 14B Q6_K | ~1.0s | ~30 |
| RTX 5060 Ti 16GB (estimated) | Qwen 3 8B Q8_0 | ~0.5s | ~50 |
| RTX 5060 Ti 16GB (estimated) | **Gemma 4 26B A4B Q4** | **~0.3-0.5s** | **~50-80** |
