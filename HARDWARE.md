# Hardware Upgrade — Serveur dédié voice assistant

## Pourquoi

Le M1 16GB fait tourner le pipeline mais le LLM (Qwen 3 4B Q4_K_M) est limité :
- ~21 tok/s, ~1.7s warm latency
- Tool calling inconsistant → beaucoup de compensation Python (room groups, text tool call parser, generic entity names, réponses TTS hardcodées)
- 8GB max pour le LLM (STT + TTS prennent le reste)

Un GPU dédié avec 16GB VRAM permettrait un modèle 14B+ qui gère tout ça nativement.

## Recommandation : Zotac Magnus EN275060TC (Barebone)

- **GPU** : RTX 5060 Ti 16GB GDDR7 (desktop, pas laptop)
- **CPU** : Intel Core Ultra 7 255HX (20 cores, 5.2 GHz turbo)
- **Format** : 2.65L (21 × 20.3 × 6.2 cm), plus petit qu'une PS5
- **Connectique** : WiFi 7, 2x 2.5GbE, Thunderbolt 4, HDMI + DP
- **Fiche produit** : https://www.zotac.com/us/product/mini_pcs/magnus-en275060tc-barebone

### À ajouter (barebone = sans RAM, SSD, OS)

| Composant | Spec | Prix estimé |
|---|---|---|
| Barebone | Zotac Magnus EN275060TC | ~1 558€ ([idealo.fr](https://www.idealo.fr/prix/208347852/zotac-zbox-magnus-en275060tc.html)) |
| RAM | 1x SO-DIMM DDR5-5600 32GB | ~80€ |
| SSD | M.2 NVMe 512GB ou 1TB | ~40-60€ |
| OS | Ubuntu Server 24.04 | gratuit |
| **Total** | | **~1 700€** |

### Ce que ça permet de faire tourner

| Modèle | Quantification | VRAM | tok/s estimé |
|---|---|---|---|
| Qwen 3 8B | Q8_0 (quasi natif) | ~9GB | ~50 tok/s |
| Qwen 3 14B | Q4_K_M | ~8.5GB | ~35 tok/s |
| Qwen 3 14B | Q6_K | ~11GB | ~30 tok/s |
| Qwen 3 32B | Q3_K_M | ~14GB | ~15 tok/s |

Le sweet spot : **Qwen 3 14B Q6_K** (~11GB, rentre large dans 16GB VRAM).

### Code Python qu'on pourrait simplifier/supprimer

Avec un 14B+ fiable sur le tool calling :
- `ROOM_GROUPS` et `resolve_all_entities()` → le modèle ferait plusieurs tool calls
- `_parse_text_tool_call()` → tool calls structurés fiables
- `_GENERIC_NAMES` → le modèle comprendrait "les volets" = tous les volets
- Réponses TTS hardcodées → le modèle génère des réponses naturelles
- `/no_think` hack → plus nécessaire avec un modèle qui gère la latence

## Alternative moins chère : Minisforum G1 Pro

- **GPU** : RTX 5060 8GB GDDR7
- **CPU** : AMD Ryzen 9 8945HX (16 cores)
- **Prix** : ~1 440€ avec 32GB RAM + 1TB SSD (prêt à l'emploi)
- **Limite** : 8GB VRAM = max Qwen 3 8B Q8_0, pas de 14B
- **Lien** : https://minisforumpc.eu/products/minisforum-atomman-g1-pro-minipc

## Benchmark comparatif (warm latency, tool calling)

| Setup | Modèle | Warm latency | tok/s |
|---|---|---|---|
| M1 16GB (actuel) | Qwen 3 4B Q4_K_M | ~1.7s | ~21 |
| M1 16GB | Qwen 3 8B Q4_K_M | ~3.1s | ~12 |
| RTX 5060 Ti 16GB (estimé) | Qwen 3 14B Q6_K | ~1.0s | ~30 |
| RTX 5060 Ti 16GB (estimé) | Qwen 3 8B Q8_0 | ~0.5s | ~50 |
