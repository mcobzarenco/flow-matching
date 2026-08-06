# Now

*Updated 2026-08-06 23:30–23:3xZ (real `date -u`) — tick (babysit):
**molmo2 AR 40k healthy at step 740/40k — loss 5.419 (5.65@540 →
5.42@740, smooth), 2.194 s/step steady (smoke bound 2.55 → ~24 h to
40k), vram_alloc_peak 66.79 GiB (rule ≤71), reserved 68.14, grad
norm 6.5–8.1, LR warming on schedule, 4 ranks pgrep-alive, util
49–99% (bursty-normal), AND THE FIRST PROBE EVAL LANDED:
eval_chunk_mae 30.844@500 (train_mae 30.71)** — no gate applies yet
(the @2500 value anchors the not-below-by-10k gate; the >25×3 gate
starts after 5k); 30.8@500 is the baseline to watch descend.
Discord: no inbound; history check caught a **🎉 reaction on our
23:04Z rc-answer + launch post** (owner celebration, recorded per
the reaction rule — no queue change). The 23:25Z A-s0 recommendation
stands unanswered → no redirect; the chained work session launches
A-s0 `draws10_t1` per the pre-reg (cost gate first ~200 frames).
Queue unchanged from 23:3xZ: **next (chained work session) → A-s0
AR draws10_t1 launch on the local GPU; then π0.5 deep-read post
(low-prio); arm A img280 HELD; molmo2 endpoint gets the same stems
at its ~08-08 boundary.** GPUs busy ×4 (box 40k) + local
idle-pending-launch + CPU queue live → `run_work_next` armed
(marker present 23:29); first save boundary @2,500 ~00:4xZ.*

*Previous update 2026-08-06 23:06–23:3xZ (real `date -u`) — work session
(bounded, conversational mid-session): **THE AR SAMPLED-DRAWS EVAL
INSTRUMENT IS LANDED + PRE-REGISTERED (ideas #19, the owner's 19:15Z
fairness ask) — the GPU-busy window's queued CPU item, delivered
whole in one session.** The build (`78c9f56`):
`--ar-temperature T --sample-draws N` temperature-samples the AR
action block N times per frame and means the decoded chunks — the
flow ensembling's mirror. Mechanics: Gumbel-max over the
grammar-masked softmax (exact masked-softmax sampling, illegal ids
can never win; aux value lines stay GREEDY), per-row CPU RNG streams
keyed by frame identity + draw (`stable_sample_rng`,
domain-separated from flow noise; corpus/batch/shard/device
invariant), draws share ONE prefill via reference cache
snapshot/restore (`ARSuffixDecoder.cache_snapshot`, sound under the
append-only cache contract, restored ≡ fresh bit-exact) — covers
Gemma AND Molmo2 trunks through the shared `ARSuffixDecoder`.
Policy row `_drawsN_tT`, `ar_temperature` in report JSON, narrated
pass skipped under sampling, loud guards everywhere; 9 CPU oracles
(T→0 limit ≡ greedy; hot draws valid/deterministic/distinct;
sampler batch-permutation invariance; mask escape impossible;
prefill-reuse bit-exactness; keying component-sensitivity + domain
separation; guard trips) — **check.py 351 green.**
[Pre-reg posted](posts/2026-08-06-prereg-ar-sampled-draws.md)
(`754f4cb`): **T=1.0 pinned/untuned as primary** (fairness rule:
flow's draws are untuned noise ⇒ AR samples its own untuned
softmax; the #19 fit-on-probe option resolved AGAINST fitting),
arms = A-s0 `_draws10_t1` (local GPU) + molmo2 AR 40k endpoint
(same stems, ~08-08), anchors = flow teacher 6.6232→5.365 / AR
greedy 5.8026, cost gate = rate-measure ~200 frames → q4-subset
fallback for BOTH arms if full-panel projects >24 GPU-h,
falsified-if Δ_AR > +0.1. Blog built + Space pushed —
**link-fix lesson: the Space serves at
`mcobzarenco-fontaine-blog.static.hf.space` (the bare `.hf.space`
domain 404s); first Discord link was wrong, corrected in-channel
23:26Z.** BABYSIT 23:23Z (molmo2 AR 40k): step 540/40k, loss
**5.653** (ahead of the smoke's 8.0@150 shape), **2.186 s/step**
live (better than the 2.55 smoke bound → ~24 h to 40k),
vram_alloc_peak 66.67 GiB FLAT (rule ≤71), reserved ~71.3 GiB
steady, grad norm 11.4, LR warming on schedule, 4 ranks alive,
util 41–100%. OWNER EXCHANGE (caught at the babysit poll, both
answered 23:25Z, conversational hold + 45-s Discord monitor since):
23:09Z "is 2.5 s per B12, i.e. 6× microbatches of 2?" → yes —
s_per_step = one optimizer step = global batch 48; each rank runs
B12 as 6 sequential 2-sample forward+backwards then the chunked
allreduce + Adam (and live it beats the smoke at 2.19); 23:20Z
"what's a good use of the local GPU while molmo2 trains?" →
recommended THIS instrument's A-s0 arm (`draws10_t1`, pre-reg
above) — **launch in the next chained work session unless the owner
redirects; any reply is steering.** Queue: **next (chained work
session) → A-s0 AR draws10_t1 launch on the local GPU per the
pre-reg (cost gate first ~200 frames); then π0.5 deep-read post
(low-prio); arm A img280 HELD (fresh owner go required); molmo2
endpoint gets the same stems at its ~08-08 boundary.** GPUs busy ×4
(box 40k, healthy) + local idle-pending-launch + CPU queue live →
`run_work_next` armed; babysits on normal cadence, K1 anchors
unchanged (launcher header + smoke shape).*

*Previous update 2026-08-06 23:03–23:1xZ (real `date -u`) — tick (babysit +
conversational): **THE MOLMO2 AR 40k IS LIVE — launched 22:57:08Z
(`fontaine_molmo2_ar_40k_ddp4`, box tmux `molmo2ar40k`, wandb
we57e8dh) and first-poll healthy: E1 banner EXACT (878 datasets /
38,571 episodes / 18,636,749 frames / dims 6/6), 4×100% util,
vram_alloc_peak 66.67 GiB (rule ≤71), 2.33 s/step at step 40 (≲28 h
to 40k), loss 16.11 → 14.46, grad norm 253 → 98, LR warming on
schedule.** This entry also back-fills the 21:0x–22:5xZ arc the
spend-cap outage swallowed (commits exist, no now.md entries): rung
5 (6×2+zero1) and rung 6 OOM'd like their predecessors → mem-snapshot
instrument built (`BIJOU_MEM_SNAPSHOT`, allocation-site attribution,
true-torch-peak per log line, 42a202a..73159c7) → rung 7 (12×1)
TRAINED but was rejected on the reserved-pool peak rule + 3.85
s/step ⇒ 43 h > F2 (1f9920b) → forensics snapshot NAMED the block:
**DDP reducer buckets, 13.6 GiB, allocated at construction — never
at sync** → rung 8 = 6×2 + zero1 + `--chunk-grad-allreduce` with NO
DDP wrapper at all (fd8bc0e, one-time param broadcast + explicit
per-step allreduce) → smoke GREEN on every gate (66.67 GiB flat,
2.52–2.55 s/step, loss 16→8.0 @150, eval + zero1 consolidated save
exercised, rc=0) → finalization cells filled (fa3048e 22:56Z) →
launch 22:57Z. HARNESS OUTAGE 22:1x–22:3xZ: monthly spend limit
(429s killed two ticks at birth + the smoke-watch session); owner
deactivated the cap 22:39Z. OWNER EXCHANGE: 22:39Z "what do you
mean by rc?" sat 24 min unanswered (the outage's tail) — answered
23:0xZ (rc = return code; it was 0) + posted the launch status +
first-poll numbers; conversational hold held ~12 min on a Discord
monitor after the reply. Kill gates (launcher header): NaN/inf;
probe not below its @2500 value by 10k; probe > 25 sustained ×3
evals after 5k — kills only at save boundaries (every 2,500, first
~00:4xZ; evals every 500). Queue: **next (chained work session) →
AR sampled-draws eval instrument (ideas #19, owner ask, separate
pre-reg — the GPU-busy window's CPU item); then π0.5 deep-read post
(low-prio); arm A img280 HELD (fresh owner go required)**. GPUs
busy ×4 (box 40k) + local idle-by-design + CPU queue live →
`run_work_next` armed; babysits on normal cadence, K1 curve
anchors = the launcher header + rung-8 smoke (loss 8.0@150 as the
early shape reference).*

*Previous update 2026-08-06 20:21–20:4xZ (real `date -u`) — tick (babysit,
held through the smoke verdict): **RUNG 4 (B12 2×6 + zero1) OOM'D AT
STEP 1'S SECOND CHUNK FORWARD — AND THE CROSS-RUNG VRAM TRACES
REWRITE THE LADDER'S MECHANISM; RUNG 5 (B12 6×2 + zero1) LAUNCHED
20:28Z.** The 19:5x–20:0x "static ~77 GiB once Adam materializes"
story was over-attributed: rung 4 died at 77.5 GiB BEFORE any
optimizer step (no step lines at log-every 20 proved nothing; the
vram trace does — 33.9 GiB init plateau → 81 GiB in ~6 s, one
monotone climb, no step structure). Measured components (traces of
rungs 1/3/4; rung 2's sampler died at 4 lines, its "step 2 once Adam
materialized" was inferred arithmetic): init static 33.9 (masters +
bf16 weights + context), activations ~2.8/sample, autocast bf16 cache
~9.7 live during each forward, DDP fp32 grads +14.6 after the first
chunk backward, Adam +29.1 unsharded at first step. ⇒ a 6-sample
chunk's forward with grads resident (48.5+9.7+~17 ≈ 75–77) OOMs in
step 1 REGARDLESS of zero1; rung 3 (6×2) genuinely completed step 1
and died at step 2 when unsharded Adam landed. **The fixes compose,
each killing exactly one block: rung 5 = 6×2 + zero1** (2-sample
chunks keep every forward in budget — proven by rung 3's step 1;
zero1 shards the Adam block that killed rung 3), predicted peak
71–73 GiB (~6 GiB margin); fallbacks 12×1, then bf16 grad buckets.
Pre-reg §3 rungs 4+5 amendments recorded pre-verdict (rung-5 TODO
cells open); launcher default flipped BACKWARD_CHUNKS 2→6; §2
plumbing line updated. Rung-4 corpse cleaned (ranks freed on their
own this time, 4×0 MiB before relaunch; failed smoke save-dir rm'd).
Discord: correction + mechanism + rung-5 note posted 20:30Z (my
20:19Z "2×6 should fit" was wrong — said so); no owner inbound this
tick (last exchange closed 20:19Z). RUNG-5 VERDICT: see the postscript
below once the boundary lands. Queue unchanged: **next (chained work
session) → rung-5 verdict → pre-reg finalization cells + launch
TONIGHT iff green (owner steer stands); then AR sampled-draws eval
instrument (ideas #19); arm A img280 HELD; π0.5 deep-read post
(low-prio)**. GPUs busy (smoke) + CPU queue deep → `run_work_next`
armed; the chained session owns the launch critical path.*

*Previous update 2026-08-06 19:59–20:0xZ (real `date -u`) — tick (babysit,
held through the smoke boundary): **SMOKE RUNG 3 (B12 × 2-sample
chunks) OOM'D AT STEP ~2 — THE CHUNK LADDER IS EXHAUSTED AND THE
MECHANISM IS NOW FULLY MEASURED: the static budget alone is ~76–77
GiB/rank, so NO chunk size fits.** Held the session through the
verdict window (monitor on the box pane): rank 0 died at a forward
RMSNorm with **77.46 GiB allocated by PyTorch** — the rung-2
arithmetic (~63 GiB static) missed the bf16 weight copy (~9.7 GiB)
+ CUDA/NCCL context. True per-rank static once Adam materializes:
bf16 weights 9.7 + fp32 masters 19.4 + DDP fp32 grad buckets 14.6 +
Adam moments 29.1 ≈ 73 + context ≈ **76–77 GiB on a 79.18 GiB card
→ ~2 GiB activation headroom**; shrinking chunks was never going to
close a static gap. Fix ranking posted to Discord (20:03Z): (1)
**ZeRO-1 optimizer sharding** (`ZeroRedundancyOptimizer` — Adam
moments 29.1 → 7.3 GiB/rank, static ~55 GiB, ~24 GiB headroom, B12
chunked 2×6 fits with margin, optimizer semantics EXACT); (2) bf16
grad buckets (halves 14.6, composable); (3) activation
checkpointing #20 (does NOT close a static gap — follow-up only).
Box cleaned this tick: hung NCCL peers torn down (rank 0 crashed,
5 peers held all 4 GPUs at 81 GiB — killed; 4×0 MiB verified),
stale `ctrl40k`/`statedrop` tmux killed (jobs long done). Discord:
no owner inbound (19:15Z cache/wandb/sampling message was answered
19:39Z; only unread was our own ftrig post); box GPUs now
idle-pending-fix, local idle. Queue: **next (chained work session)
→ ZeRO-1 (or equivalent) memory fix + re-smoke (B12 gate, same
global batch 48) + pre-reg finalization cells
(`2026-08-06-prereg-molmo2-ar-40k.md` TODO_SMOKE_*) + launch
TONIGHT iff green — the owner's molmo2-tonight steer stands; then
AR sampled-draws eval instrument (ideas #19, owner ask, separate
pre-reg); arm A img280 HELD; π0.5 deep-read post (low-prio)**.
GPUs idle-pending-fix + CPU queue deep → `run_work_next` armed;
the chained session owns the fix + launch critical path.*

*Previous update 2026-08-06 18:41–19:0xZ (real `date -u`) — work session (chained, bounded):
**MOLMO2 WP4 ASSEMBLY SLICE LANDED + THE UNTRAINED-GEN PROBE (owner
ask 18:18Z) ANSWERED SAME SESSION — the full multimodal compose
works end-to-end on the real checkpoint, and the grounding read is a
strong POSITIVE.** The build (`bijou/molmo2/model.py`): (1)
`build_multimodal_mask` — causal OR image-block, the shipped
`or_mask_function` composition re-read from `modeling_molmo2.py`
this session (any two image-typed positions mutually visible, THEN
key padding excluded); (2) `Molmo2Model` compose — additive vision
injection (`+=` at `<im_patch>` positions ONLY, count die-loud vs
the backbone's valid-token output, exactly the reference
`build_input_embeddings`), logical positions under left padding;
(3) cache-free `greedy_generate` (no KV cache exists under D1 —
probe/parity tool); (4) `load_model` full-checkpoint loader. 5 new
CPU oracles (mask vs brute-force reference semantics; additive
injection; count-mismatch die-loud; **left-pad invariance
end-to-end** — pad must leak through neither causal nor the
bidirectional image block; greedy = own argmax + stop ids) —
**check.py 327 green.** PROBE
(`fontaine/scripts/molmo2_untrained_gen.py`, local idle GPU, bf16,
real pipeline rig-frame → Collator → WP3 collator → WP4 compose,
878-id prompt / 820 image-typed at max_crops 1): raw continuation
at the training position = `[wrist camera|Image 2]<|im_end|>` — a
9-token FORMAT ECHO of our bracket syntax, NO refusal (same under
the full aux request); with the `<|im_start|>assistant` opener the
raw trunk gives an accurate scene description — "stack of wooden
coasters" = the disk stack ✓, "person holding a black electronic
device with wires" = the operator's hand on the teleop leader arm
✓, "two plastic objects" = orange boat + gripper finger ✓ — it
declines the task only because it doesn't share our naming ("toy
boat"). Second frame (15000, overexposed) same shape.
Night-and-day vs gemma4's refusals; frames posted to Discord
(composites banked `reports/molmo2_probe_frame{100,15000}.png`).
Babysits 18:42/18:57Z: masked q4 eval @992→@3,872/4,301, ~160
f/min, box GPU 0 82% util → **done ~19:0xZ, on schedule**; GPUs 1–3
idle (smoke path clear); Discord polled ×3 (boot + checkpoint +
close), no inbound. Queue: **next (chained work session) → AR
decoder arm (fast_embed + fresh head rows on the
frozen-original-vocab split per the 18:1xZ freezing answer) +
memory smoke (4.85B live trunk × 4 ranks, B32 gate) + AR 4×DDP
pre-reg + launch iff green; then ftrig ship-rule application +
arm C statedrop reads at the masked-eval boundary (~19:0xZ); arm A
img280 HELD; π0.5 deep-read post (low-prio)**. GPUs busy ×1 (box
GPU 0 finishing) + CPU queue deep → `run_work_next` armed per
no-idle-pauses; the chained session owns the eval boundary + the
AR-arm critical path.*

*Previous update 2026-08-06 18:39–18:4xZ (real `date -u`) — tick (babysit): **masked
q4 eval healthy and on schedule — @832/4,301 frames at 18:40Z (~160
f/min from the 18:35Z scan start), box GPU 0 at 82% util / 12.7 GiB,
pgrep-alive → done ~19:0xZ**, inside the predicted 19:0x–19:3xZ
window; GPUs 1–3 idle (Molmo2 smoke unblocked), local GPU
idle-by-design. Discord: no inbound; history check caught a **👍
reaction on our 18:37Z multi-view in-distribution answer** (owner
agreement, recorded per the reaction-steering rule — no queue
change). Box hygiene: killed a stale watcher loop (pid 3820072)
sleep-polling for the ctrl-eval npz under `~/flow-matching/reports/`
— the ctrl eval ran in the `~/flow-matching-ctrl` checkout and its
artifacts were rsynced local, so that path never fills. Queue
unchanged from 18:4xZ: **next (chained work session) → WP4 assembly
slice + untrained-gen probe (owner ask) + AR decoder arm + memory
smoke + AR 4×DDP pre-reg + launch iff green; then ftrig ship-rule
application + arm C statedrop reads at the masked-eval boundary
(~19:0xZ); arm A img280 HELD; π0.5 deep-read post (low-prio)**. GPUs
busy ×1 (box GPU 0) + CPU queue deep → `run_work_next` stays armed
(marker present) per no-idle-pauses; the chained session owns the
eval boundary and the tonight critical path.*

*Previous update 2026-08-06 18:15–18:4xZ (real `date -u`) — work session (bounded,
conversational mid-session): **MOLMO2 WP3 IS LANDED — the ChatML
collator + native processor are in, gated BYTE-EXACT against the
shipped trust_remote_code processor, and the FAST anchoring is
recorded in the schema** (`4113167`; the tonight critical path's
first block, owner-confirmed 18:12Z). The build: (1)
`bijou/molmo2/processor.py` — op-for-op native port of the 4.x-pinned
image pipeline (crop tiling + overlap margins + 2x2 pooling index +
token layout under the shipped options: cols on high-res rows only,
`<low_res_im_start>` marker); (2) `bijou/encoders/molmo2.py` —
`Molmo2Inputs` + `Molmo2InputsCollator`, prompt format namespaced
MOLMO2_PROMPT_FORMAT 1: images hoisted per the shipped template
bytes, `[kind camera|Image i]` bracket groups bind camera kinds to
the shipped labels, soft state token spliced inside the
`(<|im_end|>, \n)` close, LEFT padding, bos=`<|im_end|>` (checkpoint
convention), native `tokenizers` backend (segment assembly PROVEN
equivalent to whole-string tokenization); (3) golden fixtures banked
from the reference processor in its own transformers-4.57 side env
(`bank_processor_goldens.py`, 3 cases: 480p mc1, two-camera rig,
mc8 2x2 tiling) — **ids / token-type mask / grids / pooling indices
/ pixels ALL EXACT**; (4) FAST block base 152,064
(`fast_block_base`) recorded — the second extension block after the
128 image specials, embedding + fresh untied head rows
decoder-owned. Operating point `max_crops=1` → 410 image
tokens/camera (the smallest layout inside the shipped
distribution). 10 new CPU oracles, **check.py 322 green**, plan
post §6 struck through for WP3. OWNER EXCHANGE (three messages,
caught at the babysit poll + answered in-window): 18:18Z "show me
what the UNTRAINED model generates on our exact training-formatted
prompt (gemma4 gave refusals)" → acked, queued as the first
consumer of the WP4 assembly slice (it doubles as the end-to-end
prompt-path test); 18:34Z "is same-time multi-view in-distribution
given video pretraining?" → answered with template receipts
(separate image branch — `Image N` labels + 2x2 pooling, no
timestamps — and the style list's multi_image_* + mantis_instruct
training tasks ⇒ non-sequential multi-image is trained; caveat:
same-timestamp cross-VIEW binding is what the probes measure, not
assume); 18:35Z "good luck with WP4" 🍀. Box state: arm C panel
eval COMPLETED + banked ~18:2xZ (reports pulled at the boundary by
the chained session; frozen reads stay with
`statedrop_results.py`), masked q4 eval running on GPU 0
(@32/4,301 scan-warmup 18:35Z → done ~19:0x–19:3xZ), **GPUs 1–3
idle — the Molmo2 smoke is unblocked**. Local GPU idle (ftrig chain
complete; ship-rule application queued). Queue: **next (chained
work session) → WP4 assembly slice (vision injection +
bidirectional image mask + full-model compose) + untrained-gen
probe (owner ask, post generations) + AR decoder arm (fast_embed +
fresh head rows on the frozen-original-vocab split per the 18:1xZ
freezing answer) + memory smoke (4.85B live trunk × 4 ranks, B32
gate) + AR 4×DDP pre-reg + launch iff green; then ftrig ship-rule
application (likely diagnosis branch) + arm C statedrop reads at
the masked-eval boundary; arm A img280 HELD (fresh owner go
required); π0.5 deep-read post (low-prio)**. GPUs busy ×1 (box GPU
0 masked eval) + CPU queue deep → `run_work_next` armed per
no-idle-pauses.*

*Previous update 2026-08-06 18:09–18:2xZ (real `date -u`) — tick (conversational): **OWNER
STEERED THE SCHEDULE LIVE (18:10:13Z): "I want to run molmo2 tonight,
let's delay arm A" — ARM A IS HELD; the box belongs to Molmo2 AR
tonight.** The 18:1xZ plan ("arm A tonight, Molmo2 tomorrow") is
overridden at the decision point before any launch. Also answered the
owner's 18:09:30Z freezing question (both asked mid-window, caught by
the 45-s conversational poll): **for the original vocab BOTH sides
freeze** — `wte.embedding` [151,936] + shipped `wte.new_embedding`
[128 image specials] + all original-vocab `lm_head` rows (Gemma
rationale, gemma4.py:532; Qwen3's untied head makes the implicit
Gemma choice explicit) — trainable = the NEW FAST extension block
(embedding rows [152,064, 153,090) + fresh untied head rows) +
decoder layers + `ln_f`; aux-text reads the frozen shipped head,
grads flow through it into the trunk. Replied 18:12Z with the split +
the tonight plan: **critical path compresses into the chained
sessions — WP3 ChatML collator + FAST extension anchoring + AR
decoder arm + memory smoke + pre-reg, gates HELD (smoke before
launch; 4.85B live trunk × 4 ranks), pre-reg + launch tonight iff
oracles + smoke green, else the diagnosis.** Box state: arm C panel
eval @22,432/25,800 at 18:11Z (~200 f/min → panel read ~18:3xZ,
masked eval chains → GPU 0 free ~19:0x–19:3xZ; GPUs 1–3 ALREADY idle
— the smoke need not wait). Local: ftrig chain's panel-v2 forgetting
guard @19,552/22,578 at 18:10Z → chain end ~18:1xZ; after-reads +
pre-registered ship rule (rollout `--check` on @4k → upload + owner
command, or diagnosis) stay with the chained work session. Exchange continued in-window:
owner 18:12:34Z **"Agreed. Let's focus on WP3"** (acked) + 18:13:34Z
ChatML question (answered in-channel: ChatML = the
`<|im_start|>role…<|im_end|>` Qwen-family convention; Molmo2's
template puts `<|image|>` placeholders BEFORE the conversation text —
why the collator is a WP, not shared Gemma prompt code). **FTRIG
CHAIN LANDED THIS TICK — face-value after-reads (frozen ship rule
stays with the work session): rig draws1 11.4872/3.1280 (before
11.3925/3.0903), rig draws10 11.2559/3.0066 (before 10.9854/2.9126)
— NO improvement, both reads slightly WORSE; panel-v2 forgetting
guard 5.7928/1.8985 (pre-finetune 5.6711/1.7059, small drift);
state-copy rows byte-match banked (rig 12.0506/2.7702, panel
11.7639/2.5851). The in-run probe descent (13.43@500 → 12.43@2500)
did NOT convert into holdout gains — the ship rule's diagnosis
branch looks live, but the pre-registered rule decides, not this
tick.** Queue (REORDERED per steering): **next (chained work
session) → Molmo2 WP3 collator + FAST anchoring + AR decoder arm +
smoke + pre-reg (AR 4×DDP TONIGHT — top priority, owner-confirmed
18:12Z) + ftrig ship-rule application (likely diagnosis branch) +
arm C statedrop reads at its boundary; arm A img280 HELD (launcher
banked, pre-reg intact — launches only on a fresh owner go); π0.5
deep-read post + blog reorg (low-prio)**. GPUs busy ×1 (arm C box;
local freeing as the ftrig chain exits) + CPU queue deep →
`run_work_next` armed (18:06) per no-idle-pauses.*

*Previous update 2026-08-06 17:1x–18:1xZ (real `date -u`) — work session (bounded, then
extended by TWO OWNER STEERING BURSTS, conversational mode held):
**MOLMO2 WP1 + WP2 BOTH LANDED WITH FULL HF PARITY IN ONE SESSION —
the port is text+vision parity-clean and the owner has re-aimed phase 1
at AR-FIRST.** (1) WP1 (`bd5b7f9`): pure-torch Qwen3 decoder
(`bijou/molmo2/text.py` — fused-QKV GQA 32:8 hd128 1/√d, qwen3
per-head qk-norm pre-RoPE, fused-gate SwiGLU, untied embeds + 128-slot
extension matrix, residual-tap protocol identical to Gemma's),
truncated-mount loader, tiny fixture, 9 CPU oracles; the flagged
unknown died at first touch (`rope_scaling_layers: null` re-fetched).
(2) OWNER 17:24/17:27Z ("how's molmo2? check outputs vs transformers")
— **parity harness built + run same hour on CPU fp32** (221 GB RAM
takes both models; `--with transformers==4.57.1`, HF 5.x breaks the
remote code): residual stream ≤1.5e-4 all 36 layers, logits ≤3.4e-5,
greedy argmax 79/79, 15-layer mount BITWISE vs full prefix —
`PARITY PASSED`, numbers Discord'd. (3) WP2 same session (`20922af`):
`vision.py` op-for-op (25-block tower as shipped, taps [-3,-9] concat,
masked 2×2 attention pooling, gated projector), 7 more oracles (312
green), **vision parity 4.4e-7 relative on real processor inputs**
(725 image tokens; caught + documented: HF's eager path DROPS the
pooling mask — SDPA is the shipped semantics we mirror). (4) OWNER
17:51Z steering: **AR-FIRST** — deep case (their paired report
Δ−2.69@2.5k ~8× noise, hosted now under /reports/ + nav renamed
"Reports"; π0.5) → replied agreeing + proposed the FAST anchoring fix
(1,026 ids ≥ Qwen3's ~271 spare tail → SECOND trainable extension
embedding block at [152,064, 153,090) + fresh untied head rows); plan
post §6 amended (ack pending). Owner also asked re best-of-10 oracle
HTML: answered — no HTML by construction (analysis-JSON product of the
fairness instrument; offered a report page). Scheduling rec posted:
**arm A img280 still launches tonight** at arm C's boundary (no idle
box), Molmo2 AR takes the whole box TOMORROW (WP3 collator + FAST
anchoring + AR arm + smoke + pre-reg are the critical path — not
tonight, gates held). Babysits: ftrig 4k DONE ~17:50Z (loss
0.028, probe 13.43@500 → 12.43@2500 descending, K1 far), chain →
rig draws1+draws10 banked, panel-v2 forgetting guard @13,632/22,578
~1,240 f/min → **local chain end ~18:1xZ, next session owns the ftrig
after-reads + pre-registered ship rule**; arm C panel eval recovered
(60→200 f/min) @21,472/25,800 → masked eval next → **box boundary
~19:0x–19:3xZ (arm A launch per owner rec unless overridden)**. Queue:
**next (chained) → ftrig after-reads + ship rule (rollout --check on
@4k → upload + owner command, or diagnosis) + box boundary (arm A
launch runbook unchanged) + arm C statedrop reads + results post; CPU
critical path → WP3 ChatML collator + FAST extension anchoring + AR
decoder arm (owner-prioritized, AR 4×DDP pre-reg tomorrow) + π0.5
deep-read post + blog reorg (low-prio)**. GPUs busy ×2 + CPU queue
deep → `run_work_next` armed per no-idle-pauses.*

*Previous update 2026-08-06 16:4x–17:1xZ (real `date -u`) — same work session, extended by an
OWNER STEERING BURST (16:22–16:53Z, six messages — caught by the
end-of-session poll; conversational mode held since): **ALL FIVE ASKS
DELIVERED SAME SESSION.** (1) *"What's running / keep GPUs hot"* —
answered in-channel (arm C's chained evals don't create wandb runs;
box GPUs 1–3 take arm A at the boundary; local GPU re-hot within the
hour, see 4). (2) *"Upload the SnapFlow student for local NFE
rollouts + a command + make sure rollout works"* — student
`step_030000` uploaded weights-only (FIRST artifact in
`fontaine-checkpoints`), and the requested verification **caught a
real bug: the #18.2 stable noise-key default crashed EVERY flow
rollout at first predict (KeyError: repo_id — live-rig observations
have no dataset identity)**; fixed by pinning rollout to index keying
(fresh noise per replan, the historical deployment semantics) +
regression test; `bijou.rollout` gained `--target-time {t,zero}`
mirroring eval's 1-NFE switch; verified on the real student:
`--check` predict ok (50,6), cold 2.5 s, **async warm 2 ticks @30 Hz
→ SUSTAINABLE**; command posted in-channel (`63b044e`, check.py 296
green). (3) *"Standing rule: upload valuable checkpoints, no
optimizers"* — codified in charter §6 + wake-up memory; acted on
immediately per (4b). (4) *"NFE fine-tune on my rig datasets asap"* —
**PRE-REGISTERED AND LAUNCHED SAME HOUR**
([pre-reg](posts/2026-08-06-prereg-snapflow-ftrig.md), live on the
Space pre-launch): `fontaine_flow_snapdistill_ftrig_4k_1xh100` —
student-verbatim recipe + 5 deltas (rig-only data, init-from student,
4k steps, LR 1e-5, save 500), `--distill snapflow` CONTINUED so the
shortcut field adapts with the velocity field; **R0 before-reads
banked first** (rig holdout 3,647 frames, 1-NFE stable: student
draws1 **11.3925/3.0903**, draws10 10.9854/2.9126 vs state-copy
12.0506/2.7702 — the un-tuned student barely beats copy on chunk and
LOSES on first_mae on rig; the transfer gap is the whole case for
this run); E1 banner exact (2 datasets/51 episodes/32,431 frames/
dims 6/6, strict student load), first-poll 100% util, 22.5 GiB,
0.49–0.51 s/step (E2 band), loss ~0.05 → **4k ~17:4xZ, chained
after-reads (rig draws 1/10 + panel-v2 1-NFE forgetting guard) land
~18:0x–18:3xZ.** (4b) *"Review all previous checkpoints, upload the
valuable ones"* — inventory done, verdicts posted: ALL FIVE box AR
40k endpoints uploaded weights-only (s0/s1/s2/auxoff/statedrop;
6 runs now on the hub, optimizer-free verified); intermediates prune
at boundaries, smoke checkpoints deleted, not uploaded. (5)
*"How does statedrop work?"* — answered in-channel (mean-masking at
collation via the shared `mask_state_item` primitive — train dropout
and eval `--mask-state` can never drift; owner 👍). MEANWHILE arm C's
panel eval @10,752/25,800 at 16:56Z (~213 f/min) → panel read
~18:1xZ, masked eval after → **box boundary (arm A launch) ~18:4x–
19:1xZ**. Queue: **next session → babysit ftrig (K1: probe > first
read + 3.0 ×3 evals ≥1.5k) + box boundary when arm C's chain ends
(live-checkout pull → pytest → arm A launch per the 16:4x entry's
runbook) → ftrig after-reads + ship rule (rollout --check on @4k →
upload + owner command, or the diagnosis) → arm C statedrop reads +
results post**; CPU next → Molmo2 WP1. GPUs busy ×2 (ftrig local,
arm C eval box GPU 0) + CPU queue deep → `run_work_next` armed per
no-idle-pauses.*

*Previous update 2026-08-06 16:04–16:4xZ (real `date -u`) — work session (chained, bounded):
**ARM C TRAINING COMPLETE AT 40k (~16:02Z) AND THE ARCH-BATCH LAUNCH
PATH IS FULLY DE-RISKED — launchers banked, recipes machine-verified,
F1 SMOKES BOTH GREEN AT B32 — arm A launches at the arm-C eval-chain
boundary with zero improvisation left.** (1) Arm C
(`fontaine_arb_rcond_statedrop80_40k_1xh100`) finished its 40k
schedule: final loss 3.6583, LR 1e-5 on schedule, formal final probe
**10.8961@40000 — the launcher's pre-registered final gate "probe <
10 @40k" FAILS by 0.90** (record-only here: p=0.8 may be too
aggressive at this rung; the informative reads are the chained panel
eval + masked q4 reliance eval, frozen assembly in
`statedrop_results.py`). Chained panel eval live on GPU 0
(first-poll: 60–76% util, 12.6 GiB, warming 40→140+ f/min,
2,592/25,800 @16:25Z) → panel read ~18:3x–19:1xZ, masked eval after
→ **chain end ~19:0x–20:0xZ, that session owns the launch + reads.**
(2) Arm A/B launchers + F1 smoke banked (`f382629`):
teacher-verbatim recipe verified through the REAL
`bijou.train.parse_args` vs the banked teacher@40k train_args
(`fontaine/scripts/arch_recipe_verify.py`, 56 fields verbatim + 8
pre-registered deltas per arm) — the verify CAUGHT a real drift
pre-launch (teacher trained `--prompt-generate-bracket`; first
launcher draft omitted it, fixed); chained panel-v2 endpoint evals
pinned heun30/draws1/`--noise-key stable` with stems per
`arch_batch_results.py`. (3) **F1 MEMORY SMOKE: BOTH ARMS PASS AT
B32** — run this session from the HEAD-updated `~/flow-matching-ctrl`
throwaway on GPUs 1–3 (arm C's eval owns GPU 0; live checkout
untouched per never-sync-under-live-run; box-side pytest 295 green in
the ctrl checkout first): arm A img280 peak **22,825 MiB**,
0.87–0.91 s/step (F2: 40k ≈ **10 h** ≪ 30 h gate → full 40k, no
screen-rung); arm B fullresid peak **26,705 MiB**, 0.54 s/step (~6 h;
res-adapter banner 498.1M decoder params, grads flowing, loss curve
tracks arm A's). B32 stands for both arms; smoke rates are
upper bounds (measured under arm-C-eval CPU contention). Class fix:
smoke verdict lines now tee into the log (first run's echoes died
with the tmux pane; recomputed from the banked vram sampler logs,
pulled local to `reports/smoke_arch*`). Lit slice taken (~15 min):
IVRA (arXiv:2601.16207) banked into ideas #15 — training-free
single-LM-layer patch-affinity injection, fits the #11 acuity-probe
diagnosis; rung-(a) candidate if arm A leaves grounding headroom.
Discord: no inbound ×3 polls (16:04/16:14 boot+babysit, 16:4x end).
Queue: **boundary session (~19:0x–20:0xZ, chained/tick-owned) → live
checkout `git pull --ff-only` to ≥`f382629` + box pytest re-verify →
launch arm A via `~/launch_box_gpu123_fontaine_flow_archA_img280_40k_ddp3.sh`
(tmux, first-poll util+rate vs smoke 0.88 s/step; K1 babysits via
`arch_batch_results.py --k1-train-log`) → pull arm C reports →
`statedrop_results.py` frozen reads + results post → cleanup
(`~/flow-matching-ctrl`, `outputs/train/smoke_arch*`, stale
`~/launch_local_snapflow_distill*` copies, `~/smoke_arch_ctrl.sh`)**;
CPU next → Molmo2 WP1 Qwen3 decoder port (`bijou/molmo2/text.py`,
plan §3). GPU busy ×1 (arm C panel eval, box GPU 0) + CPU queue deep
→ `run_work_next` armed per no-idle-pauses.*

*Previous update 2026-08-06 15:43–16:0xZ (real `date -u`) — work session (chained, bounded):
**TWO DELIVERABLES — (1) #18.2 `--noise-key` DEFAULT FLIPPED to
`stable` (`d77ed58`): the hold expired when the SnapFlow chain's
index-keyed stage-4 evals + npz addendum completed — CLI +
`BijouPolicy`/`SmolVLAEvalPolicy` ctors now default `stable`, `index`
retained permanently behind an explicit flag for historical
reproduction, new default-pin regression test, check.py 295 green;
arm A/B launchers written at the box boundary inherit `stable` as the
arch-batch pre-reg requires. (2) OWNER STEERING 15:49Z (mid-session
Discord poll caught it 6 s after posting — the class fix earning its
keep): eval reports linked from posts + hosted on the Space —
DELIVERED SAME SESSION (`bbafaee`): all 21 banked HTML eval reports +
9 frozen analysis JSONs uploaded to the blog Space under `/reports/`
(154 MB, curl-verified 200), new [reports index](reports.md) page in
the nav grouping every report by run, inline report links added to 6
results posts (SnapFlow ×3 endpoint reports, stable-key re-bank,
box-batch, draws-fairness ×2, state-probe, flow-vs-AR), blog built +
Space pushed, Discord headline with direct links posted 15:58Z; owner
👍'd the plan reply.** Arm C babysits 15:50/15:55Z: @38,500→39,060/40k,
TRAIN-ALIVE (pgrep 2), 66–78% util, 73.8 GiB, 0.374–0.384 s/step,
loss 3.59–3.74 smooth, aux 0.50–0.57, LR decayed to 1.01e-5 exactly
on schedule → **40k ~16:1xZ — the next chained session owns the
boundary: code sync + stage-0 re-verify + F1 two-config smoke → arm A
img280 launch + arm C statedrop reads + `~/flow-matching-ctrl`
cleanup.** Local GPU idle-by-design. Discord: owner inbound 15:49Z
(the reports ask, answered + delivered); no further inbound through
16:0xZ. Queue: **box boundary (~16:1xZ, next session) → the launch
sequence above; CPU next → Molmo2 WP1 Qwen3 decoder port
(`bijou/molmo2/text.py`, plan §3, top CPU item)**. GPU busy ×1 (arm C
box) + CPU queue deep → `run_work_next` armed per no-idle-pauses.*

*Previous update 2026-08-06 15:41–15:4xZ (real `date -u`) — tick (babysit): **arm C
inside the final 2.5k — @37,620/40k at 15:41Z, TRAIN-ALIVE (pgrep 2),
65% util (eval window), 73.8 GiB, 0.37–0.39 s/step, loss 3.58–3.74
smooth, aux 0.43–0.54, in-run probe holding the descended band:
10.85@36000 → 10.96@36500 → 10.83@37000 → 11.00@37500** (vs the
11.1–11.58 plateau @34–35.5k; K1 margin far), zero substitution lines
→ **40k ~16:1x–16:2xZ**. Local GPU idle-by-design. Discord: no owner
inbound (only unread was our own 15:40Z WP0 headline;
history-checked, no new reactions) — **the SnapFlow adoption ask
stays open; any reply is steering.** Queue unchanged from 15:4xZ:
**next (chained) → #18.2 `--noise-key` default-flip + Molmo2 WP1
Qwen3 decoder port (top CPU item)**; box boundary (~16:1x–16:2xZ) →
code sync + stage-0 re-verify + F1 two-config smoke → arm A img280
launch + arm C statedrop reads + `~/flow-matching-ctrl` cleanup. GPU
busy ×1 (arm C box) + CPU queue deep → `run_work_next` armed (marker
present, 15:40) per no-idle-pauses; the chained session owns the 40k
boundary.*

*Previous update 2026-08-06 15:23–15:4xZ (real `date -u`) — work session (chained, bounded):
**MOLMO2 WP0 IS LANDED — the port's first work package, one session
after the plan, exactly as sequenced (`7409df0`): the trunk seam is
extracted.** The build: `ObservationEncoder[I: BatchInputs, B:
nn.Module]` ABC now lives at the seam (`bijou/interface.py` —
stream_geometries / inputs_collator / encode / param_groups, the
`docs/plan.md` contract with the backbone passed by the composition
root); `KVCache` is OUT of the seam (`ObservationMemory.cache` is
opaque — trunk-private contract, ar_backbone isinstance-narrows);
`BijouModel[I, B]` + the whole train-loop surface
(DevicePrefetcher/ChunkedBatch/ChunkingCollator/ProbeSet/
build_probe_set/validate/BijouTrainStep) de-Gemma-typed, Gemma-only
paths (ar_backbone suffix continuation, tensor-level
encode_observation, save_checkpoint's prompt schema) narrow LOUDLY;
`PromptKind.MOLMO2` reserved beside GEMMA4 with a refuse-until-WP4
loader arm (+ test). Impl-time decision the plan left open:
`StreamGeometry` grows NO `scaling` field — under D1 the adapters
absorb scale, it would be dead config with one legal value. Gates
held exactly as pre-declared: **check.py 294 green (was 293 + the
new MOLMO2 refusal test), three CPU loss oracles bit-exact, no
state-dict key changes, pyright clean.** Plan post §6 sequence
struck-through for WP0 (blog built + Space pushed); **WP1 (Qwen3
decoder port) is now the top CPU item.** Arm C babysits 15:24/15:38Z:
@35,820→37,160/40k, TRAIN-ALIVE (pgrep 2), 73–99% util, 73.8 GiB,
0.374–0.39 s/step, loss 3.61–3.73 smooth, aux 0.45–0.49, and the
in-run probe DESCENDED below the plateau band — **10.85@36000 →
10.96@36500 → 10.83@37000** (vs 11.1–11.58@34–35.5k; K1 margin far)
→ 40k ~16:1x–16:3xZ. Local GPU idle-by-design. Discord: no inbound
×2 polls (boot + 15:38Z babysit; history-checked) — **the owner
adoption ask from the SnapFlow results post is still open; any reply
is steering.** Queue: **next (chained) → #18.2 `--noise-key`
default-flip (small, unblocked) + Molmo2 WP1 decoder port
(`bijou/molmo2/text.py`, plan §3)**; box boundary (~16:1x–16:3xZ) →
code sync + stage-0 re-verify + F1 two-config smoke → arm A img280
launch + arm C statedrop reads + `~/flow-matching-ctrl` cleanup. GPU
busy ×1 (arm C box) + CPU queue deep → `run_work_next` armed per
no-idle-pauses.*

*Previous update 2026-08-06 15:22–15:2xZ (real `date -u`) — tick (babysit): **arm C healthy
and inside the final 5k — @35,540/40k at 15:23Z, TRAIN-ALIVE (pgrep),
73.8 GiB, 68% util, 0.38 s/step, loss 3.70–3.81 smooth, aux
0.47–0.51, probe 11.3304@35500 (11.11–11.58 plateau band @34–35.5k
unchanged, K1 margin far), zero substitution lines → 40k ~16:3xZ.**
Local GPU idle-by-design (SnapFlow arc complete and published).
Discord: no inbound, no new reactions (history-checked; last 5 are
our own posts through the 15:18Z results headline) — **the owner
adoption ask from the results post is still open; any reply is
steering and outranks the queue.** Queue unchanged from 15:3xZ:
**next (chained) → Molmo2 WP0 seam refactor (top CPU item) + #18.2
`--noise-key` default-flip (unblocked)**; box boundary (~16:3xZ) →
code sync + stage-0 re-verify + F1 two-config smoke → arm A img280
launch + arm C statedrop reads + `~/flow-matching-ctrl` cleanup. GPU
busy ×1 (arm C box) + CPU queue deep → `run_work_next` armed (marker
present, 15:20) per no-idle-pauses.*

*Previous update 2026-08-06 15:13–15:3xZ (real `date -u`) — work session (chained, bounded): **THE
SNAPFLOW RESULTS POST IS PUBLISHED — the #12 distillation arc closes
public: parity-adopt + deployment headline, live on the Space,
Discord'd with the owner adoption ask**
([results](posts/2026-08-06-snapflow-results.md)). All six TODO
cells filled from the frozen analysis JSON (no judgment at write
time): draws5 row 5.3918/1.6056; diversity read — mean-of-5 banks
~90% of the student's total averaging gain (same fractional shape as
the teacher's 87%, at **one-fifth the amplitude**, 0.236 vs 1.258),
student mean-of-5 already beats teacher mean-of-5 (5.3918 vs
5.5235), mean10 0.03 below the modal band per the report's
`inside_modal_band: false`; per-step horizon read — student below
teacher at ALL 50 steps (`crossover_step: null`), delta widening
monotonically −0.229@1 → −1.554@50, i.e. distillation compressed
late-horizon error hardest (mean-collapse operating where draw
spread is largest — the 08-05 flow-vs-AR divergence does NOT
transfer to student-vs-teacher); v2 descriptive column
5.6711/1.7059 vs teacher 6.7151/1.9453 — margin −1.04,
keying-robust; npz addendum ~27 min (14:43→15:10Z); adoption cell
quotes the instrument verbatim + the concrete proposal (single draw
= latency floor at 1 expert eval, mean-of-10 = quality mode at ~10,
still 3× under one Heun-30 draw). Final s=t 7.6601 VERIFIED in the
train log before shipping (step-30000 line). `check.py` 293 green →
commit 119e12e → blog built + Space pushed → post URL live with
draws5 numbers curl-verified → **Discord headline + owner adoption
ask posted 15:2xZ**. Arm C babysit 15:1xZ: @35,000/40k, pgrep
alive, 73.8 GiB, 0.373–0.385 s/step, probe **11.1057@35000**
(11.11–11.58@34–35k, plateau band unchanged, K1 margin far) → 40k
~16:3xZ; the 0%-util instant was the step-35000 eval/save boundary,
rate on-band. Discord: no inbound ×2 polls (boot + post-publish).
Queue: **next (chained) → Molmo2 WP0 seam refactor (top CPU item) +
#18.2 `--noise-key` default-flip (NOW UNBLOCKED — the SnapFlow chain
and addendum are complete)**; box boundary (~16:3xZ) → code sync +
stage-0 re-verify + F1 two-config smoke → arm A img280 launch + arm
C statedrop reads + `~/flow-matching-ctrl` cleanup. Owner watch
item: the adoption decision requested in-channel — any reply is
steering. GPU busy ×1 (arm C box; local idle-by-design) + CPU queue
deep → `run_work_next` armed per no-idle-pauses.*

*Previous update 2026-08-06 15:11–15:2xZ (real `date -u`) — tick (babysit): **THE SNAPFLOW
FROZEN VERDICT IS IN — `snapflow_results.py` ran this tick on the
complete input set (npz addendum banked 15:0xZ, local GPU idle at
poll) and the pre-registered decision assembly fires PARITY /
ADOPT-SIGNAL + DEPLOYMENT HEADLINE**
(`reports/analysis__snapflow_distill_30k_k4l2.json`): read 1 primary
1-NFE chunk 5.6036 ≤ adopt line 6.7732 (Δ vs teacher −1.02); read 2
grounding edge survives (first 1.7039 ≤ 1.9831, teacher 1.9331);
read 3 deployment headline holds — mean-of-10 5.3675/1.5927 ≤ AR
anchor 5.8026 (draws5 5.3918/1.6056; teacher draws10-heun30
5.3645/1.4242 — student matches chunk to 3 dp, first lags ~0.17;
mean10 sits 0.03 BELOW the modal band [5.4, 5.6], flagged
`inside_modal_band: false`, consistent with the flat draw-averaging
curve = mean-collapse shape rather than surviving diversity — the
INTERPRETATION cell belongs to the results post, instrument banked
face-value only). The verdict's own adoption line: the charter §2
cost caveat on the draws win closes — mean-of-10 @1-NFE costs ~10
expert evals, not 10×30 Heun steps → **results post + owner adoption
decision are the deliverable**, skeleton already staged
(`posts/2026-08-06-snapflow-results.md`, TODO cells). Arm C
@34,500/40k at 15:13Z, TRAIN-ALIVE (pgrep 2), 65% util, 73.8 GiB,
0.374–0.381 s/step, loss 3.68–3.69 smooth, aux 0.46–0.48, probe
**11.16@34000 (11.10–11.43 band @32.5–34k)** — plateau band
unchanged, K1 margin far → 40k ~16:3xZ. Local GPU idle-by-design
(SnapFlow GPU arc complete; next local GPU work only via a new
pre-reg). Discord: no inbound, no new reactions (history-checked;
last 5 are our own posts through the 14:07Z Molmo2-plan headline).
Queue: **next session (chained) → fill + publish the SnapFlow
results post (verdict cells from the analysis JSON, draw-averaging
interpretation, blog build + Space push + Discord headline) → Molmo2
WP0 seam refactor (top CPU item)**; box boundary (~16:3xZ) → code
sync + stage-0 re-verify + F1 two-config smoke → arm A img280 launch
+ arm C statedrop reads + `~/flow-matching-ctrl` cleanup. GPU busy
×1 (arm C box) + CPU queue deep → `run_work_next` armed per
no-idle-pauses.*

*Previous update 2026-08-06 14:42–15:0xZ (real `date -u`) — tick (babysit): **ALL THREE
SnapFlow endpoint evals are in — draws5 landed 14:39Z at
5.3918/1.6056 (record-only, v1 panel index keying; state-copy rows
byte-match banked 11.7848/2.6202) — and the draw-averaging curve is
nearly FLAT: 5.6036 (draws1) → 5.3918 (draws5) → 5.3675 (draws10),
a gain of only ~0.24 from 10× averaging vs the teacher's ~1.26
(6.6232 → 5.365).** Face-value shape (NO frozen interpretation —
that's `snapflow_results.py`'s job): consistent with the 1-NFE
student's draws being far less diverse than the teacher's — each
draw already sits near the posterior mean, i.e. the mean-collapse
hypothesis from the fairness work, not surviving draw diversity.
Local GPU was idle at 14:42Z (draws5 chain complete) → **addendum
npz eval launched this tick** (tmux `npzaddendum`,
`eval_snapdistill_endpoint_1nfe_npz.sh`, quiet-GPU guard passed,
`--noise-key index` pinned): scoring @8,672/25,800 at 15:00Z,
measured 1,280 f/min over a 45-s window (bursty-util-normal, 9.3
GiB) → done ~15:1xZ. Arm C @31,500/40k at 14:43Z, 0.37–0.376
s/step, TRAIN-ALIVE (pgrep 1), 63% util, 73.8 GiB, loss 3.72–3.85
smooth, aux 0.43–0.62, in-run probe **11.18@30000 → 11.42@30500 →
11.53@31000** (band-bouncing 11.2–11.5 after 11.26@27500 — plateau
region, not a spike; K1 margin far) → 40k ~16:3x–17:3xZ unchanged.
Discord: no inbound, no new reactions (history-checked; last 5 are
our own posts through the 14:07Z Molmo2-plan headline). NOTE: a
~14:23–14:27Z chained session pre-staged the results-post skeleton
(`posts/2026-08-06-snapflow-results.md`, draws5/verdict cells TODO)
+ SUMMARY.md line but ended without committing or a now.md entry —
skeleton committed with this tick (git-only; blog NOT rebuilt, the
TODO skeleton is not reader-ready). Queue: **next session (chained)
→ npz addendum boundary (~15:1xZ) → `snapflow_results.py` frozen
reads (all inputs now exist) → fill + publish the results post →
Molmo2 WP0 seam refactor (top CPU item)**; box boundary
(~16:3x–17:3xZ) → code sync + stage-0 re-verify + F1 two-config
smoke → arm A img280 launch + arm C statedrop reads +
`~/flow-matching-ctrl` cleanup. GPUs busy ×2 (npz addendum local,
arm C box) + CPU queue deep → `run_work_next` armed per
no-idle-pauses.*

*Previous update 2026-08-06 14:09–14:1xZ (real `date -u`) — tick (babysit): **draws10
landed AT the boundary — held the session through the handoff and the
draws5 chain kicked over clean.** draws10 endpoint eval completed
~14:12Z: **mean-of-10 chunk 5.3675 / first 1.5927** (record-only, v1
panel index keying; state-copy rows byte-match banked 11.7848/2.6202)
— BELOW the AR deployment anchor 5.8026 and matching the teacher's own
full-panel mean-of-10 (5.365) to 3 dp, i.e. the 1-NFE student's
10-draw deployment read is teacher-equivalent at face value; whether
that reflects surviving draw diversity or mean-collapse is exactly
what the draws5 read + `snapflow_results.py` frozen assembly will say
— NO interpretation banked here. draws5 chained automatically (log
14:12Z, dataset-scan warming; ~25,800 frames at ~1,000 f/min → done
~14:4xZ, then all three endpoint evals are in). Arm C @27,500/40k at
14:12Z, 69% util, 73.8 GiB, TRAIN-ALIVE, in-run probe
**11.2568@27500** (12.09@27000 → 11.26 — descending through the
12.0–12.4 band, K1 margin far) → 40k ~16:3x–17:3xZ unchanged.
Discord: no inbound, no new reactions (history-checked; last 5 are our
own posts through the 14:07Z Molmo2-plan headline). Queue unchanged
from 14:1xZ: **next session → draws5 finish → addendum npz eval
(`eval_snapdistill_endpoint_1nfe_npz.sh`) → `snapflow_results.py`
frozen reads + results post → Molmo2 WP0 seam refactor (top CPU
item)**; box boundary (~16:3x–17:3xZ) → code sync + stage-0 re-verify
+ F1 two-config smoke → arm A img280 launch + arm C statedrop reads +
`~/flow-matching-ctrl` cleanup. GPUs busy ×2 (draws5 local, arm C box)
+ CPU queue deep → `run_work_next` stays armed (marker present, 14:08)
per no-idle-pauses.*

*Previous update 2026-08-06 13:55–14:1xZ (real `date -u`) — work session (chained, bounded):
**THE MOLMO2-4B PORT PLAN IS POSTED — the owner's 12:03Z "get started
in the background" steer now has its first deliverable, plan before
code as asked**
([plan](posts/2026-08-06-molmo2-port-plan.md), primary sources
fetched this session and distilled into `docs/molmo2.md` per the §6
post-cutoff rule — `config.json`, `preprocessor_config.json`,
`chat_template.jinja`, `modeling_molmo2.py` all read today, plus a
full bijou code-surface audit with file:line receipts). The design
calls: **residual-only conditioning** (arm B's path — learned
adapters keep the expert contract at kv1×512 regardless of Qwen3's
GQA 32:8; no KVCache/layer-type/`project_kv` port) and a
**15-of-36-layer mount** (fractional depth 0.417 vs E2B's mounted
15/35 = 0.429 — expert depth and the res0..res14 schedule carry over
unchanged; SmolVLA/FLOWER early-layers support banked yesterday).
One Qwen3-4B decoder port amortizes across Molmo2-4B /
InternVL3.5-4B (the #10 base-vs-IT vehicle) / Qwen3-VL-4B. Phase 1
trains flow on the **raw frozen prefix** — no AR port, no vocab
surgery (flow never touches FAST ids); the AR-adaptation −2.7
confound ships with any claim, comparison declared vs a matched
raw-Gemma-prefix baseline. Five WPs (seam refactor → decoder port +
HF parity → SigLIP tower → ChatML collator → schema/audit), §4
oracle suite gates any pre-reg; mounted footprint ~2.3B ≈ 4.7 GiB
bf16; est. 4–6 CPU sessions — background work for the GPU-busy
windows, exactly as promoted. `check.py` 293 green; blog built +
Space pushed (post URL 200); Discord headline posted. Babysits
13:55/14:01/14:07Z: draws10 @21,632/25,800 (~1,000 f/min sustained,
bursty-util-normal) → done ~14:1xZ, draws5 chains → all three
endpoint evals land ~14:3x–14:4xZ; arm C @26,500/40k, 0.37–0.40
s/step, loss 3.73–3.95 smooth, aux 0.48–0.63 → 40k ~16:3x–17:3xZ
unchanged. Discord: no inbound ×3 polls. Queue: **next session →
draws10/draws5 boundary → addendum npz eval
(`eval_snapdistill_endpoint_1nfe_npz.sh`) → `snapflow_results.py`
frozen reads + results post**; box boundary (~16:3x–17:3xZ) → code
sync + stage-0 re-verify + F1 two-config smoke → arm A img280
launch (eval names per instrument stems) + arm C statedrop reads +
`~/flow-matching-ctrl` cleanup; CPU next → **Molmo2 WP0 seam
refactor** (now the top CPU item; plan §6 sequence) + dataset dedup
script/manifest + #16 follow-ups + #18.2 default-flip (after the
chain); ≥2 ✓. GPUs busy ×2 (draws evals local, arm C box) + CPU
queue deep → `run_work_next` armed per no-idle-pauses.*

*Previous update 2026-08-06 13:50–13:5xZ (real `date -u`) — tick (babysit): **both
GPUs healthy; SnapFlow's draws10 endpoint eval is past a fifth of the
panel and running hot.** draws10 @5,632/25,800 frames at 13:52Z,
~1,100 f/min measured over a 90-s window (73–82% util) → done ~14:1xZ,
draws5 chains automatically after → **all three endpoint evals landed
~14:3x–14:4xZ; the chained work session owns the addendum npz eval +
`snapflow_results.py` frozen reads + results post.** Arm C @26,040/40k
at 13:52Z, 0.371 s/step train, 72% util, 73.8 GiB, loss 3.88–3.95
smooth, aux 0.59–0.65, in-run probe **12.02–12.40@24.5–25.5k** (vs
12.68@20000 — descending band, K1 margin far) → 40k ~16:3x–17:3xZ
unchanged (box boundary: code sync + stage-0 re-verify + F1 two-config
smoke → arm A img280 launch per instrument stems + statedrop reads +
`~/flow-matching-ctrl` cleanup). Discord: no inbound (the one unread
was our own 13:49Z work-session headline), no new reactions
(history-checked). Queue unchanged from 13:5xZ. GPUs busy ×2 (draws
evals local, arm C box) + CPU queue deep → `run_work_next` armed per
no-idle-pauses.*

*Previous update 2026-08-06 13:12–13:5xZ (real `date -u`) — work session (chained, bounded):
**ARCH-BATCH RESULTS INSTRUMENT BANKED BEFORE ANY DATA (oracle-before-
data, 5th application) — AND THE CONTROL EVAL LANDED THROUGH IT THE
SAME SESSION: teacher@40k = 7.1041/2.0720 on panel-v2 stable, INSIDE
the Amendment 1 expectation band [6.7, 7.9]/[1.90, 2.35].**
`fontaine/scripts/arch_batch_results.py` (f7c3238) encodes every
frozen read of the [pre-reg](posts/2026-08-06-prereg-arch-batch-1.md)
+ Amendments 1–2: paired per-frame Δchunk/Δfirst vs control with
seeded frame-bootstrap CI95, adopt ≤−0.15 / grounding ≤−0.10 /
falsified >+0.15 classification (sub-band class for measurable-but-
under-floor), read-4 assembly (both-null ⇒ Molmo2-4B promotion; B
adopt ⇒ upstream offer; A move ⇒ 560 rung justified), strict
endpoint-semantics guards (heun30/draws1/stable/panel-v2
15,056+7,522 rows die-loud), and the K1 kill gate vs the teacher's
banked in-run probe curve (`reports/teacher_artrunk40k_probe_curve.
json`, pulled from the box train log, 9.1306@5000 verified —
babysits at arm launches run `--k1-train-log`). Five oracles green
incl. v2 anchor reproduction 6.7151/1.9453 + state-copy
11.7639/2.5851 through this file's OWN keep-mask semantics; 7 new
CPU tests; `check.py` 293 green. Ctrl eval finished 13:47Z on box
GPU 1 (throwaway checkout), artifacts rsynced local, `--ctrl-only`
read + report cross-check OK — **arm A can launch at the box
boundary with zero improvisation left on the reads side** (arm
launchers MUST name endpoint evals per the instrument stems:
`…archA_img280…/…archB_fullresid…__step_040000__panel_v2_heun30_
draws1_stable`). MEANWHILE SNAPFLOW CROSSED 30k (13:14Z) and its
**primary 1-NFE endpoint eval completed 13:42Z — RECORD-ONLY:
chunk 5.6036 / first 1.7039 (v1 panel, index keying), past the
6.7732 adopt line by 1.17 and BELOW both the teacher's own Heun-30
(6.6232/1.9331) and the AR anchor (5.8026); state-copy rows
byte-match banked (11.7848/2.6202)** — consistent with the fairness
finding (1-NFE endpoint ≈ posterior mean; chunk MAE rewards mode
non-commitment); the draws10/draws5 evals now running will show
whether draw diversity survived distillation; frozen reads stay
with `snapflow_results.py` once all three land (+ addendum npz
eval at the chain boundary). Lit slice (~15 min, sanctioned):
SmolVLA (~L/2 conditioning) + FLOWER (prunes 50% deep layers)
banked as independent support for the early-layers story — arm B's
cheap follow-on if it nulls is an EARLY-ONLY schedule, not more
layers; VLM4VLA (ICLR 26 survey): downstream VLA perf uncorrelated
with VLM benchmarks ⇒ the Molmo2-4B port plan's case must rest on
vision-tower/grounding properties, not benchmark rank (ideas #11,
d826a2f). Queue-debt cleared: the mid-session Discord-poll class
fix is ALREADY in both prompts (verified work.md + tick.md) —
struck from the queue. Babysits 13:12/13:28/13:42Z: arm C
@23,860/40k, 0.391 s/step, loss 4.03, aux 0.51 descending → 40k
~16:3x–17:3xZ; SnapFlow draws10 eval warming. Discord: no inbound
×3 polls. Queue: **next session → SnapFlow draws10/draws5 babysit →
addendum npz eval (`eval_snapdistill_endpoint_1nfe_npz.sh`) →
`snapflow_results.py` frozen reads + results post**; box boundary
(~16:3x–17:3xZ) → code sync + stage-0 re-verify + F1 two-config
smoke → **arm A img280 launch (eval names per instrument stems)** +
arm C statedrop reads + `~/flow-matching-ctrl` cleanup + stale
`~/launch_local_snapflow_distill…sh` copy cleanup; CPU next →
Molmo2-4B port plan (owner-promoted) + dataset dedup
script/manifest + #16 follow-ups + #18.2 default-flip (after the
chain); ≥2 ✓. GPUs busy ×2 (SnapFlow draws local, arm C box; box
GPU 1 freed 13:47Z) + CPU queue deep → `run_work_next` armed per
no-idle-pauses.*

*Previous update 2026-08-06 13:03–13:2xZ (real `date -u`) — tick (conversational): **OWNER
STEERED ARM A LIVE (12:59Z, two messages): 560 soft tokens is too many
for 480p sources, try 280 — ADOPTED, Amendment 2 posted + live on the
pre-reg** ([pre-reg](posts/2026-08-06-prereg-arch-batch-1.md)). Replied
13:03Z with the agreeing pixel math (640×480 native; the processor
upscales to hit budgets — 280 ≈ 1.4× linear / ~33×33 native px per
token, 560 ≈ 2× / ~23×23, so 560's marginal tokens are the most
interpolated; 280 also halves wall cost ~12–16 h and shrinks the
FLOPs-vs-grounding confound). **Arm A := `fontaine_flow_archA_img280_
40k_ddp3` (`--max-soft-tokens 280`); 560 demoted to a follow-on rung
contingent on a positive 280 read; F2 chain = 280 → 10k-screen; cost
~25–35 GPU-h.** Blog built + Space pushed (Amendment 2 verified live).
DISCOVERY on the box: the prior session had already launched the
**teacher@40k control eval at 13:02:41Z** (tmux `ctrl40k`) — its first
attempt crashed at plan-load (bcbf101's SamplePlan refuses version-2
plans, `.crashed-1303Z` preserved), relaunch runs from the throwaway
`~/flow-matching-ctrl` checkout at HEAD (59dac60 fix; live checkout
untouched under arm C per never-sync-under-live-run); healthy
@672/22,578 frames at 13:09Z, 99% util, ~100 f/min warming → done
~14:x–16:xZ; its uncommitted launcher edits are committed with this
tick. SnapFlow @29,520/30k at 13:10Z (0.48–0.51 s/step, loss
0.035–0.039) → **30k ~13:14Z, stage-4 endpoint evals chain
automatically (1-NFE euler-1: draws 1 primary, then 10, 5; index
keying by design — #18.2 flip stays parked until they finish)**. Arm C
@22,000/40k at 13:10Z, 0.38–0.40 s/step, 94% util, loss 3.93–4.02,
aux 0.53–0.61 descending, probe 12.56–13.37@21–22k band (vs
12.68@20000; not catastrophic, K1 margin far) → 40k ~16:3x–17:3xZ.
Queue: **chained work session (armed) → SnapFlow endpoint-eval babysit
+ addendum npz eval + `snapflow_results.py` frozen reads + ctrl-eval
babysit/reads**; CPU next → `arch_batch_results.py` instrument +
Molmo2-4B port plan (owner-promoted) + dataset dedup script/manifest +
#16 follow-ups + #18.2 default-flip (after the chain) + mid-session
Discord-poll class fix; box boundary (~16:3x–17:3xZ) → code sync +
stage-0 re-verify + **F1 two-config smoke at 280** + arm A img280
launch + arm C statedrop reads + `~/flow-matching-ctrl` cleanup; ≥2 ✓.
GPUs busy ×3 (SnapFlow local, arm C + ctrl eval box) + CPU queue deep
→ `run_work_next` armed per no-idle-pauses; conversational window held
past the owner's 12:59Z messages (replies + amendment posted, no
further inbound through 13:2xZ).*

*Previous update 2026-08-06 12:49–12:5xZ (real `date -u`) — tick (babysit): **both runs
healthy; SnapFlow's 30k boundary is ~25 min out.** SnapFlow @27,300/30k
at 12:50Z, 99% util, 0.48–0.52 s/step, loss 0.038–0.041 → **30k +
chained endpoint evals ~13:15–13:3xZ** (next session owns the boundary:
endpoint evals + addendum npz eval + `snapflow_results.py` frozen reads
+ teacher@40k control eval on idle box GPU 1). Arm C @20,020/40k
(halfway) at 12:50Z, 73.8 GiB, 62% util (eval window), loss 4.02
smooth, aux 0.61 descending, **in-run probe 12.6835@20000** (16.64@8500
→ 12.68@20000, monotone descending) → 40k ~16:3x–17:3xZ unchanged; box
GPUs 1–3 idle as planned (GPU 1 reserved for the control eval next
session). Discord: no inbound, no new reactions (history-checked; last
5 are our own posts through the 12:48Z link fix). Queue unchanged from
12:5xZ: **next session (~13:1xZ) → SnapFlow 30k endpoint evals +
addendum npz eval + frozen reads + teacher@40k control eval**; CPU next
→ `arch_batch_results.py` instrument + Molmo2-4B port plan
(owner-promoted) + dataset dedup script/manifest + #16 follow-ups +
#18.2 default-flip (after the chain) + mid-session Discord-poll class
fix; box boundary (~16:3x–17:3xZ) → code sync + stage-0 re-verify + F1
two-config smoke + arm A launch + arm C statedrop reads; ≥2 ✓. GPUs
busy + CPU queue deep → `run_work_next` armed per no-idle-pauses.*

*Previous update 2026-08-06 12:12–12:5xZ (real `date -u`) — work session (bounded): **ARM B'S
RESIDUAL-STREAMS IMPLEMENTATION LANDED WITH ALL FIVE PRE-LAUNCH ORACLES
GREEN — the F1 critical path is clear** (the two-config memory smoke
needs arm B's config to EXIST, so arm A could not launch before this;
[pre-reg](posts/2026-08-06-prereg-arch-batch-1.md) pre-launch gate).
The build: `--conditioning-streams residual` in `bijou.train` — the
encoder exports raw post-layer hidden states res0..res14
(`ObservationMemory.residuals`, new `TextModel` tap API), and the flow
expert projects them through learned per-layer adapters that mirror
`TextAttention.project_kv` byte-for-byte in convention (RMSNorm →
bias-free K/V proj → learned k_norm / scale-less v_norm → keys RoPE'd
at logical positions) so the produced streams are contract-identical
to K/V exports and the SuffixBlocks are UNTOUCHED. Key design point:
adapters live DECODER-side (`expert.safetensors`,
`res_adapters.res{i}.*`) and attach OUTSIDE the no-grad prefix encode
(`FlowDecoder.attach_residual_streams`, called once per observation in
`BijouModel.encode`/`encode_observation`) — trainable under the frozen
trunk, once-per-observation cost at eval. Real-config count **23.62M
adapter params ✓** (pre-reg ≈23.6M; kv_heads 1 × head_dim 512, schedule
res0..res14 1:1 ascending). Oracles (i)–(v) as 11 CPU tests
(`tests/test_residual_streams.py`): stream contract + padding-
orientation invariance (the kv streams' own gate, applied to the
adapters), trunk bitwise-frozen through a real optimizer step, grads
reach every adapter param (adaRMS zero-gate-at-init subtlety caught
and documented — perturbed heads test the PATH), checkpoint round-trip
with no flags + strict weight load, K/V path untouched (no adapter
keys in kv-mode state_dicts, no raw taps in kv-mode memories; legacy
checkpoints load via setdefault back-compat). **`check.py` 285
green** (three CPU loss oracles included, bit-exact). SCHEDULE
CORRECTION vs the 12:05Z queue: F1 + arm A launch move from ~13:1xZ to
the **arm-C 40k boundary (~16:3x–17:3xZ)** — F1's drop-together rule
smokes BOTH configs before ANY arm launches, and arm B's code syncs to
the box only at that boundary (never under live arm C); the 13:1xZ
session should instead run the **teacher@40k control eval on idle box
GPU 1** (pre-registered, box code bcbf101 suffices — stable keying
predates it; panel-v2 plan JSON is a data push, not code). Owed at the
boundary sync: box stage-0 re-verify + F1 two-config smoke → arm A
launch. Babysits 12:12/12:37Z: SnapFlow @24,440→26,020/30k, 0.48–0.52
s/step, loss ~0.038–0.042 → **30k + chained endpoint evals
~13:1x–13:3xZ**; arm C @17,920/40k, 0.375–0.378 s/step, 83% util, loss
4.03–4.11 smooth, aux 0.62 descending → 40k ~16:3x–17:3xZ. Discord: no
inbound. Queue: **next session (~13:1xZ) → SnapFlow 30k endpoint evals
+ addendum npz eval + `snapflow_results.py` frozen reads + teacher@40k
control eval on box GPU 1**; CPU next → `arch_batch_results.py`
instrument (oracle-before-data, 5th application) + **Molmo2-4B port
plan (owner-promoted)** + dataset dedup script/manifest + #16
follow-ups + #18.2 default-flip (after the chain) + mid-session
Discord-poll prompt fix (class debt); box boundary (~16:3x–17:3xZ) →
code sync + stage-0 re-verify + F1 smoke + arm A launch + arm C
statedrop reads; ≥2 ✓. GPUs busy (SnapFlow local, arm C box) + CPU
queue deep → `run_work_next` armed per no-idle-pauses.*

*Previous update 2026-08-06 12:05–12:2xZ (real `date -u`) — tick (conversational): **OWNER
STEERED THE ARCH BATCH LIVE (12:02–12:05Z, three messages) — ARM 0 IS
DROPPED; Amendment 1 posted + live on the pre-reg**
([pre-reg](posts/2026-08-06-prereg-arch-batch-1.md)). The steering:
(1) 12:02Z agreed on arms A (img560) + B (full-residual); (2) 12:03Z
**"get started on Molmo2-4B in the background too … quite an involved
implementation piece"** → the port is PROMOTED to the CPU queue now,
independent of the batch verdict (first deliverable: port plan —
processor/tokenizer/vision-tower mapping, stream-export points, memory
budget — posted before any code; both-null branch rule unchanged);
(3) 12:05:46Z "on arm 0, we have a good enough control in ar 100k …
what do you think?" → answered 12:09Z: AR-100k is cross-family (wrong
control for ±0.15 flow-side paired reads; rides along as context row),
but the instinct is right — **control := teacher's own `step_040000`,
VERIFIED on the box before amending: completed 40k schedule
(train_args.steps=40000, LR decayed to 1e-5 at 40k; step_080000 was a
resume-extension), seed 0 MATCHED to arms (the pre-reg's "teacher used
seed 1" line was wrong — seed 1 is the SnapFlow run; struck), eff-96
(2×48 vs 3×32, topology-only), in-run probe curve banked (168 evals,
9.1306@5000) → K1 re-anchors to it at matched steps.** One panel-v2
control eval (~1–2 GPU-h) replaces the 8–10 GPU-h arm-0 retrain; F1
smoke shrinks to two configs; cost ~25–40 GPU-h, ~1–1.5 d wall; launch
order A → B. Amendment 1 live (blog built + Space pushed, URL 200);
NOTE: the 12:02/12:03Z owner messages had been cursor-consumed by the
prior session's 12:03Z poll but never recorded — recovered via
`history` (the standing history-check earned its keep). Babysits
12:05/12:13Z: SnapFlow @22,900/30k, 0.48–0.51 s/step, loss ~0.04, s=t
7.5233@22500 (flat band, kill line 9.6755 far) → **30k ~13:0xZ +
chained endpoint evals ~13:1x–13:3xZ**; arm C @16,640/40k, 0.373–0.377
s/step, 70% util, 73.8 GiB, loss 4.27 smooth, aux 0.69 descending →
40k ~16:3x–17:3xZ. Queue: **next session (~13:1xZ) → SnapFlow 30k
endpoint evals + addendum npz eval + `snapflow_results.py` frozen
reads, THEN F1 two-config smoke + teacher@40k control eval + arm A
launch on GPUs 1–3 (owner look satisfied — steering received and
amended, no further wait)**; CPU next → arm-B residual-streams impl +
5 oracles + `arch_batch_results.py` instrument + **Molmo2-4B port plan
(NEW, owner-promoted)** + dataset dedup script/manifest + #16
follow-ups + #18.2 default-flip (after the chain) + mid-session
Discord-poll prompt fix (class debt); box → arm C 40k → statedrop
reads; ≥2 ✓. GPUs busy + CPU queue deep → `run_work_next` armed;
conversational window held ~10 min past the owner's last message.*

*Previous update 2026-08-06 11:49–12:0xZ (real `date -u`) — work session: **ARCHITECTURE
BATCH #1 PRE-REGISTERED — the owner's 11:44Z multi-GPU steering is now
a posted pre-reg, Discord'd for a look before launch**
([pre-reg](posts/2026-08-06-prereg-arch-batch-1.md)). Design: 3
sequential DDP3 runs on box GPUs 1–3, stage-2 family (flow h1024
adaRMS expert on the FROZEN `bijou_arb_rcond_100k_ddp4/step_100000`
trunk — best lineage, conditioning-side is where both levers live),
40k steps, B32/rank ×3 = eff-96 (teacher-matched), panel-v2 + stable
keying (first pre-reg under both adoptions), seed 0 all arms: **arm 0**
`fontaine_flow_arch0_base_40k_ddp3` (teacher recipe verbatim,
own-baseline), **arm A** `…archA_img560…` (`--max-soft-tokens 560` —
processor-native rung MEASURED today on the real processor: budgets
{70,140,280,560,1120}, patches scale exactly linearly; 480p-upscale
caveat stated; fallback 280 on the 30-h rate gate), **arm B**
`…archB_fullresid…` (res0..res14 hidden-state streams, learned
per-layer K/V projections ≈23.6M params replace kv4/9/14; impl + 5
oracles owed pre-launch, code syncs to box ONLY at arm C's boundary).
Frozen reads: paired vs arm 0, adopt-lever iff Δchunk ≤ −0.15
CI95-excl-0, grounding iff Δfirst ≤ −0.10; both-null promotes the
Molmo2-4B trunk swap to the next multi-GPU pre-reg. Gates: F1 3-config
memory smoke before arm 0 (OOM ⇒ whole batch drops batch together,
never per-arm), F2 arm-A rate, K1 probe > arm0+3.0 @≥5k. Cost ~35–50
GPU-h, ~1.5–2 d wall; explore class (≥20% budget). Blog built + Space
pushed (post URL 200), SUMMARY.md line added, ideas #11/#17 updated,
`check.py` 274 green. Babysits 11:49/12:03Z: SnapFlow @22,100/30k,
0.48 s/step, loss ~0.043 → **30k + chained endpoint evals ~13:1x–13:3xZ**;
arm C @16,000/40k, 0.375 s/step, loss 4.28, aux 0.63 → 40k
~16:3x–17:3xZ. Discord: no owner inbound ×3 polls (11:49/11:52/12:03Z
— mid-session polls honored per the owed class fix). Queue: **next
session (~13:1xZ boundary) → SnapFlow 30k endpoint evals + addendum
npz eval + `snapflow_results.py` frozen reads, THEN F1 smoke + arm 0
launch on GPUs 1–3 (owner look window ≈1.5 h by then; no steer =
proceed per the 11:46Z exchange)**; CPU next → arm-B residual-streams
impl + oracles + `arch_batch_results.py` instrument (oracle-before-
data, 5th application) + dataset dedup script/manifest + #16
follow-ups + #18.2 default-flip (after the chain) + mid-session
Discord-poll prompt fix (class debt); box → arm C 40k → statedrop
reads; ≥2 ✓. GPUs busy + CPU queue deep → `run_work_next` armed;
sleep-poll skipped (owner quiet since 11:44Z, >10-min window closed —
ticks own the channel per the boot contract).*

*Previous update 2026-08-06 11:44–11:5xZ (real `date -u`) — tick: **OWNER REPLIED
AT 11:44:18Z (4 s before the poll) — THE 5-DECISION LIST IS ANSWERED +
NEW STEERING: a multi-GPU run aimed at FUNDAMENTAL ARCHITECTURE CHANGES
(owner examples: new trunk / full residual streams rather than few
exported layers / bigger images = more tokens per image — "really just
examples").** Decisions: (1) E4B **paused** (owner floated smaller
batch/no-accum as alternative — answered in-channel: batch-independent
floor ~110 GiB/rank, Adam m/v never allocated at peak 81.0/81.6; ZeRO-1
is the real lever, queued behind the architecture run); (2) **panel-v2
ADOPTED for all new pre-regs** + owner asks: dedup the whole dataset and
upload to hub? — replied: yes, as a versioned NEW dataset
(`community_curated_v1_dedup`, no overwrite), dedup script + manifest
staged as a CPU item, deltas posted before upload, one paired train arm
before it becomes default recipe; (3) **ES ADOPTED as diagnostic
column** — "why never headline" answered (deployment consumes one draw;
AR's ES degenerates to ~MAE so cross-family ES headlines flatter the
stochastic family); (4) stage-2b not addressed → stays parked behind
#11 per our rec; (5) GPUs 1–3 tenancy SUPERSEDED by the architecture
ask. Explained #11 in-channel (grounding front; owner's examples ARE
#11, bolder) and proposed: **next pre-reg = paired arms (a) bigger
images/more visual tokens + (b) full-residual conditioning, same trunk,
DDP on box GPUs 1–3, panel-v2; trunk swap (Molmo2-4B) as its own
follow-on; pre-reg posted for a look before launch** — owner asked for
preference-or-proceed, proceeding with 1+2 unless steered. Babysits
11:44–11:46Z: SnapFlow @20,160/30k, loss ~0.04, 0.48–0.50 s/step → 30k
+ chained endpoint evals ~13:2x–13:3xZ unchanged; arm C @14,620/40k,
0.374–0.377 s/step, 72% util, 73.8 GiB, loss 4.28 smooth, aux 0.60
descending → 40k ~16:3x–17:3xZ unchanged. Queue (NEW ORDER, steering-
driven): CPU next → **architecture pre-reg draft (arms a+b, THE
work item)** + **dataset dedup script/manifest (staged, deltas before
upload)** + #16 follow-ups + #18.2 default-flip (after the chain) +
mid-session Discord-poll prompt fix (class debt); local → SnapFlow 30k
boundary ~13:2x–13:3xZ → addendum npz eval → `snapflow_results.py`
frozen reads; box → arm C 40k ~16:3x–17:3xZ → statedrop reads; box
GPUs 1–3 → architecture run once pre-reg posted (+ owner look). ≥2 ✓.
GPUs busy + CPU queue deep → `run_work_next` armed; conversational
mode held ~10 min past last owner message per the boot contract.*

*Previous update 2026-08-06 11:2x–11:4xZ (same work session, extended) — **OWNER
STEERING CAUGHT LATE AND ANSWERED: two 10:15–10:16Z messages (GPU-
utilization question + request for a pending-decisions list + overnight
summary) surfaced only at the 11:25Z end-of-session poll — this session
held the harness lock through the probe boundary, so no tick polled the
channel in between. Class fix owed: long work sessions must poll
Discord at every babysit checkpoint, not only at boot/end (prompt edit
queued as CPU debt).** Replied 11:26Z in three structured posts: (1)
honest utilization answer — local ~98%, box GPU 0 busy (arm C), box
GPUs 1–3 mostly idle since the E4B no-launch, ~42% box aggregate since
17Z; cause = the E4B follow-on decision outstanding + anti-goal rule,
not missing ideas. (2) **Five pending decisions posted with recs: E4B
follow-on (rec: drop → #11 grounding arms, ZeRO-1 queued), panel-v2
adoption (rec: adopt for new pre-regs), ES column (rec: adopt as
diagnostic), stage-2b (rec: park behind #11), GPUs 1–3 tenancy (3× #11
arms or arm-C seed replicates, ~1 h from a go).** (3) Overnight
high-level summary. Sleep-polled ~15 min after replying (11:26–11:42Z),
no owner response yet — handed back to ticks per the ~10-min-silence
rule. **NEXT SESSIONS: owner replies to the decision list are the top
watch item — any answer is steering and outranks the queue.***

*Previous update 2026-08-06 09:4x–10:5xZ (real `date -u`) — work session: **TWO
DELIVERABLES — (1) #18.5 RIG-ROLLOUT SAFETY GATE LANDED (the
first-physical-run blocker, deep-dive findings 8+9): new lerobot-free
`bijou/rollout_safety.py` wired into `bijou.rollout` — clamp mandatory
(`--max-relative-target` positive/finite or the arm does not move;
`--unclamped` explicit opt-out; contradiction dies), first-obs
envelope assert (per-joint q01..q99 half-band-widened, 15° floor,
mean±3σ fallback; wrong stats / ticks-vs-degrees / uncalibrated arm
die loud with a per-joint table; dim≠6 = wrong-embodiment),
camera kinds now MIRROR TRAINING (`annotation_stamp` +
`camera_kinds_of` via `--stats-dataset`; unstamped/hash-mismatch →
"unknown" exactly as training rendered; `--camera-kind NAME=KIND`
validated override; the name heuristic survives only datasetless) —
22 new CPU tests, `--check` exercised on the real flow-80k checkpoint,
`check.py` 274 green (`e95b9ef`). (2) SNAPFLOW @10k 1-NFE PROBE READ
(record-only, pre-registered): chunk_mae 5.9222 / first_mae 1.8193 on
the stride-7 subset — kill line 9.6755 passed by 3.75, and the 1-NFE
distill at ONE-THIRD training BEATS the teacher's own Heun-30 read
(6.676/1.928, same frames, pairing certified by state-copy rows to 4
dp). The s=t drift (8.03@10000, flat 7.8–8.4 band) is DECONFIRMED as
a 1-NFE proxy — it measures the velocity mode, not the one-step
mode.** Probe ran on box GPU 1 (expert-only 1.8G push — teacher
backbone already on-box byte-identical, sha256-verified; box code
bcbf101 has the 1-NFE switch, no code sync under live arm C; one
relaunch after a tmux-PATH miss, ~20 min wall total); artifacts pulled
local (`reports/eval__snapdistill__step_010000__probe_s7_1nfe_euler1.json`).
Lit slice TAKEN (~15 min, debt cleared): one-step fallback menu banked
into #12 — OFP self-distillation, MeanFlow-VLA (2603.01469, kills the
consistency constraint), "Let It Be Simple" (2606.05737, high-noise
training alone). Babysits through the session: SnapFlow @11,500+/30k,
~0.49 s/step, loss ~0.04, s=t 8.17@11500 → 30k + chained endpoint
evals ~13:2x–13:3xZ unchanged; arm C @8,500+/40k, ~0.37 s/step train,
in-run probe 24.05@4500 → 22.29@5000 → **16.64@8500** descending,
40k ~16:3x–17:3xZ unchanged. Discord: no inbound. Queue: local →
SnapFlow babysit → 30k + chained endpoint evals ~13:2x–13:3xZ →
addendum npz eval → `snapflow_results.py` frozen reads (endpoint
adopt-signal ≤ 6.7732 now LIKELY on the probe prior); box → arm C
babysit → 40k ~16:3x–17:3xZ → statedrop reads ~19:xZ–21:xZ (box
step_010000 staging cleanable at that boundary); CPU next: **#16
follow-ups + #18.2 default-flip (after the chain)**; ≥2 ✓. GPUs busy
(SnapFlow local, arm C box) + CPU queue non-empty → `run_work_next`
armed per no-idle-pauses; the chained session babysits to the 30k
boundary.*

*Previous update 2026-08-06 09:35–09:3xZ (real `date -u`) — tick (babysit): **both
runs healthy; the SnapFlow s=t drift has FLATTENED — 8.3344@5000 →
8.3609@5500, +0.03 over the last 500 steps vs +0.40 the window before,
sitting ~1.76 over the teacher anchor 6.5997** (record-only; the
informative read stays the @10k 1-NFE probe, kill line 9.6755).
SnapFlow @5,500/30k at 09:35Z, 100% util, 0.475–0.51 s/step, loss
~0.042, grad norm ~0.3 → **10k probe boundary ~10:1x–10:2xZ
unchanged** (quiet-GPU decision: box GPUs 1–3 idle, checkpoint push
standing option), 30k ~13:2x–13:3xZ. Arm C @5,000/40k at 09:36Z,
0.371–0.376 s/step, 66% util (eval window), 72.5 GiB, loss 4.91
smooth, aux 0.82, in-run probe 27.78@3500 → 27.17@4000 →
**24.05@4500** descending — 40k ~16:3x–17:3xZ unchanged. Discord: no
inbound (the one unread was our own 09:35Z work-session headline;
history-checked, no new reactions). Queue unchanged: local → SnapFlow
babysit (s=t watch; 10k probe decision ~10:1x–10:2xZ) → 30k + chained
endpoint evals ~13:2x–13:3xZ → addendum npz eval →
`snapflow_results.py` reads; box → arm C babysit → 40k ~16:3x–17:3xZ
→ statedrop reads ~19:xZ–21:xZ; CPU next: **#18.5 rig-rollout safety
gate** + #16 follow-ups + #18.2 default-flip (after the chain); ≥2 ✓.
GPUs busy (SnapFlow local, arm C box) + CPU queue non-empty →
`run_work_next` armed per no-idle-pauses; the chained session takes
the 10k-probe decision and the next CPU item.*

*Previous update 2026-08-06 09:13–09:4xZ (real `date -u`) — work session: **SNAPFLOW'S
ENDPOINT RESULTS INSTRUMENT IS BANKED BEFORE ITS DATA — the box-batch
oracle-before-data pattern, FOURTH consecutive application —
`fontaine/scripts/snapflow_results.py` encodes every frozen read of the
[SnapFlow pre-reg](posts/2026-08-06-prereg-snapflow-distill.md) +
Amendment 1: the @10k probe kill line (teacher probe 6.6755 + 3.0 =
9.6755, strictly >), endpoint adopt-signal (1-NFE chunk ≤ 6.7732),
falsification (> 7.1232), grounding edge (first ≤ 1.9831), deployment
headline (mean-of-10 ≤ 5.8026, modal band [5.4, 5.6]), the per-step
horizon read (flow_vs_ar_paired protocol — oracled to byte-match its
banked curve) and the panel-v2 descriptive column (reproduces
6.7151/1.9453 from the teacher npz).** Banking early caught a REAL gap:
the running launcher's chained stage-4 endpoint evals dump JSON+HTML
only — no npz — so the pre-reg's promised per-step read had NO data
source; editing a live bash script is unsafe, so the addendum
`eval_snapdistill_endpoint_1nfe_npz.sh` (staged, quiet-GPU-guarded,
`--noise-key index` pinned explicitly per the d9dd385 lesson) re-runs
the primary with `--dump-predictions` at the boundary after the chain
(~30–40 min). Strict semantics guards refuse doctored endpoint JSONs
(steps≠1 / heun / target_time t / stable keying / draws mismatch /
subset-as-panel all die loud); oracles (a)–(e) all green on banked data
with zero SnapFlow endpoint bytes in existence; 8 new CPU tests;
**`check.py` 252 green** (`4d48120`). STANDING NOTE: `bijou.eval`'s
`--noise-key` default must stay `index` until the SnapFlow chain's
stage-4 evals execute at 30k (they inherit the default at run time and
the registered comparators are index-keyed); the default flip is #18.2
follow-on debt for AFTER the chain. Babysits 09:13/09:31Z: SnapFlow
@5,000/30k, 100% util, ~0.48 s/step, step_005000 saved → **10k probe
boundary ~10:1x–10:2xZ**; s=t divergence **7.74@2000 → 7.12@4000 →
7.93@4500 → 8.33@5000 — drifting up ~1.7 above the teacher anchor,
record-only but now the top watch item** (kill line 9.6755 is probe-@10k
1-NFE, not this s=t read; SnapFlow's claim is endpoint parity, mid-run
drift is in-model for consistency training — the 10k probe is the
informative read). Arm C @4,760/40k, 0.373 s/step, 86% util, loss 4.98
smooth, aux 0.82, in-run probe 27.8→27.2→24.1 descending — 40k
~16:3x–17:3xZ unchanged. Discord: no inbound ×2 polls. Queue: local →
SnapFlow babysit (s=t watch; 10k probe decision ~10:1x–10:2xZ — box
GPUs 1–3 idle, checkpoint push standing option) → 30k + chained
endpoint evals ~13:2x–13:3xZ → addendum npz eval → `snapflow_results.py`
reads; box → arm C babysit → 40k ~16:3x–17:3xZ → statedrop reads
~19:xZ–21:xZ; CPU next: **#18.5 rig-rollout safety gate** + #16
follow-ups + #18.2 default-flip (after the chain); ≥2 ✓. GPUs busy
(SnapFlow local, arm C box) + CPU queue non-empty → `run_work_next`
armed per no-idle-pauses; the chained session takes the 10k-probe
decision and the next CPU item.*

*Previous update 2026-08-06 09:0x–09:1xZ (real `date -u`) — tick (babysit): **both runs
healthy; ARM C'S 40k BOUNDARY RE-PROJECTED ~16:3x–17:3xZ (was ~12:3x–12:4xZ) —
the 0.37 s/step projection ignored the ~4-min in-run eval probes.** Measured
directly: arm C trains at 0.378 s/step but each 500-step eval costs ~3.7 min
wall (step-3000 eval timed 09:04:44→~09:08:2x), effective ~1.0–1.15 s/step so
far — and A-s0's own checkpoint history confirms this is SIBLING-NORMAL, not a
slowdown (17:15Z→01:17Z, ~8 h for the identical 40k + 80 evals; its first 5k
also ran ~1.06 s/step effective before averaging down to 0.72). Arm C
@3,100/40k at 09:09Z, loss 5.25 smooth (21.7→5.25), aux ~0.9, 65–68% util
during eval, 72.5 GiB; in-run evals 45.2→31.6→34.2→28.7→28.8 by step 2500 —
in-family with A-s0 at the same steps (24.3 @2500; the gap is the expected
p=0.8 masking difficulty), descending. Panel + masked reads move ~15:3xZ →
~19:xZ–21:xZ. SnapFlow local @2,780/30k at 09:11Z, 0.43–0.51 s/step, ~0.55
effective incl. evals+saves (measured 09:06→09:11: 620 steps/333 s over an
eval+save boundary — SnapFlow's evals are cheap, unlike the AR arms'), loss
~0.04, grad norm ~0.3; **s=t divergence: 7.5682@500 → 6.9939@1000 →
7.6521@1500 → 7.7359@2000** — oscillating ~1σ-ish above the teacher anchor
6.5997, record-only, far from the teacher-probe+3.0 kill line. Revised: 30k
~13:2x–13:3xZ (SnapFlow now finishes FIRST, no longer co-timed with arm C);
**step_010000 1-NFE probe boundary ~10:2xZ** — the chained session takes the
quiet-GPU decision (box GPUs 1–3 idle, checkpoint push standing option).
Discord: no inbound, no new reactions (history-checked; last 5 are our own
posts through 08:32Z). Queue unchanged: local → SnapFlow babysit (s=t watch;
10k probe ~10:2xZ) → 30k + endpoint reads ~13:3xZ+; box → arm C babysit → 40k
~16:3x–17:3xZ → panel + masked reads through the banked instrument
~19:xZ–21:xZ; CPU next: **#16 rig-transfer follow-ups** + #18 debt; ≥2 ✓.
GPUs busy (SnapFlow local, arm C box) + CPU queue non-empty →
`run_work_next` armed per no-idle-pauses; the chained session takes the
10k-probe decision and the next CPU item.*

*Previous update 2026-08-06 08:5x–09:1xZ (real `date -u`) — work session (chained):
**ARM C'S RESULTS INSTRUMENT IS BANKED BEFORE ITS DATA —
`fontaine/scripts/statedrop_results.py` encodes all three frozen reads
of the [state-dropout pre-reg](posts/2026-08-06-prereg-state-dropout-40k.md)
plus the E3 probe gate and the full verdict assembly (adopt-default /
free-hardening-lever / mechanism-inert-kill / p=0.3-screen branch /
falsified), oracled on the banked A-s0 npz with zero arm-C bytes in
existence.** The box-batch pattern, third time: oracle (a) anchor
reproduction through this file's own pooling (A-s0 7.7966/3.9422,
panel state-copy 11.7848/2.6202, q4-subset state-copy first 2.4316 —
all three sibling-instrument semantics confirmed byte-compatible);
(b) degenerate C:=A-s0 → read 1 exactly 0 / CI [0,0] and the
neutral-adopt path composes; (c) synthetic known effects — 1.05×
error inflation → COSTS verdict at the exact predicted +0.05×frame-MAE
delta with the p=0.3 branch, 0.95× → HELPS, 6.2× masked inflation →
capability "failed" ≥15 → the pre-declared MECHANISM-INERT kill, 1.5×
→ strong+sanity → hardening-lever adoption, probe-final 10.5 → E3
gate blocks all adoption; (d) misaligned masked index → hard abort.
4 new CPU tests (`tests/test_statedrop_results.py` — capability
boundaries, every verdict branch incl. the inclusive band edges,
known-delta and degenerate analyze math); **`check.py` 244 green.**
The ~12:3x–12:4xZ arm-C boundary is now zero-improvisation: defaults
point at the chained eval's output names, `--probe-final` takes the
train log's last in-run probe. Babysits 08:53/09:03Z: SnapFlow
@1,800/30k, 0.47–0.48 s/step, 87–98% util, loss ~0.043; **in-run s=t
divergence: 7.5682@500 → 6.9939@1000 → 7.6521@1500** — oscillating
around the teacher's level (stable-key anchor 6.5997), record-only,
far from the teacher-probe+3.0 kill line. Arm C @2,760/40k, 0.37
s/step, 73% util, loss 21.7→5.38 smooth, aux ~0.9. Discord: no
inbound ×2 polls. Queue: local → SnapFlow babysit (s=t watch; **10k
record-only 1-NFE probe ~10:0xZ needs a quiet GPU — box GPUs 1–3
idle, checkpoint push is the standing option**) → endpoint reads
~13:xZ; box → arm C babysit → 40k boundary → panel + masked reads
through the NEW instrument ~15:3xZ (pulled earlier at 0.37 s/step:
~12:4xZ + evals); CPU next: **#16 rig-transfer follow-ups** + #18
debt; ≥2 ✓. GPUs busy (SnapFlow local, arm C box) + CPU queue
non-empty → `run_work_next` armed per no-idle-pauses; the chained
session takes the 10k-probe decision and the next CPU item.*

*Previous update 2026-08-06 08:37–08:5xZ (real `date -u`) — tick (babysit): **SNAPFLOW
DISTILL IS TRAINING — both launch gates passed and the run is FAR ahead
of budget: ~0.49 s/step steady → 30k lands ~12:5xZ, not the 12–20 h
estimate.** Held the session through the fresh-launch critical window:
gate (b) drift eval completed (2,458 frames, step0-extended vs banked
flow npz frame-MAE drift **0.01451** < 0.05 → GATE (b) PASSED; gate (a)
had re-passed at launch), train started 08:43Z (`train_fontaine_flow_
snapdistill_h1024_30k_1xh100.log`), first-poll rule met at steady state
(92–100% util, 0.491–0.503 s/step, 22.4 GiB; distill loss ~0.038 flat
with tiny grad norms — expected from identity init under warmup), and
the **first in-run s=t divergence reading landed: eval_chunk_mae 7.5682
@step 500** (record-only watch; the only kill line is the catastrophic
teacher-probe+3.0). Revised timeline: step_010000 (the record-only
1-NFE probe boundary, needs a quiet GPU) ~10:0xZ, step_030000 + chained
endpoint evals ~12:5x–13:xZ — nearly co-timed with box arm C's 40k
boundary. Box arm C healthy on its second poll: step 1,480/40k at
08:38Z, 0.374 s/step, 77% util, 69.8 GiB, loss 21.7→5.65 smooth →
step_040000 ~12:3x–12:4xZ, panel + masked-reliance reads after.
Discord: no inbound, no new reactions (history-checked; last 5 are our
own posts through the 08:32Z re-bank+SnapFlow headline). Queue
unchanged: local → SnapFlow babysit (s=t watch; 10k probe when a quiet
GPU appears — box GPUs 1–3 are idle, checkpoint push is an option) →
endpoint reads ~13:xZ; box → arm C babysit → 40k boundary → panel +
masked reads ~15:3xZ→ pulled EARLIER if arm C holds 0.374 s/step
(~12:4xZ + evals); CPU next: **arm-C results instrument prep (oracle
before data, box-batch pattern)** + #16 follow-ups; ≥2 ✓. GPUs busy
(SnapFlow local, arm C box) + CPU queue non-empty → `run_work_next`
armed per no-idle-pauses; the chained session takes the results
instrument and the 10k-probe decision.*

*Previous update 2026-08-06 07:51–08:4xZ (real `date -u`) — work session (chained):
**THREE MOVES IN ONE SESSION — #9 STATE-DROPOUT ARM PRE-REGISTERED AND
LAUNCHED ON THE IDLE BOX; #18.2 STABLE-KEY RE-BANK ADOPTED (flow anchor
6.5997); SNAPFLOW DISTILL LAUNCHED ON THE FREED LOCAL GPU.** (1) The
state-probe branch rule cashed in: `--state-dropout` landed
(`bcbf101` — shared `mask_state_item` primitive with the eval probe so
semantics can never drift; p=0 bitwise-inert, all three CPU loss
oracles exact, `check.py` 240 green, SnapFlow stage-0 re-verified),
[pre-reg posted](posts/2026-08-06-prereg-state-dropout-40k.md), then
**arm C `fontaine_arb_rcond_statedrop80_40k_1xh100` launched 08:10Z on
box GPU 0** (idle since the E4B no-launch; A-s0 recipe verbatim +
`--state-dropout 0.8`, seed 0, 40k; E1 selection lines byte-match
A-s0's, banner `p=0.8` ✓; first-poll: 91% util, 66.3 GiB, 0.386–0.395
s/step steady — E2 met; loss 21.7→6.5 by step 500; step_040000
~12:2xZ, chained panel + masked-reliance evals land reads ~15:3xZ).
(2) **Flip re-bank ADOPTED at the 08:30Z boundary**
([results](posts/2026-08-06-stablekey-rebank-results.md)): controls
bitwise ✓, stable-key chunk **6.5997** inside [6.4882, 6.7582]
(Δ −0.0242 ≈ 1σ_draw), first 1.9355; `stable` is now the quoted
keying for all new flow numbers, ledger re-banked, #18.2 closed.
(3) **SnapFlow distill launched 08:30Z** (tmux `snapdistill`) on the
GPU the flip freed — after a pre-launch catch: the launcher had
inherited the TEACHER's `bijou-dev` wandb project (READ-ONLY mainline,
§7) + `bijou_` name prefix via the teacher-verbatim copy; fixed to
`fontaine`/`fontaine_flow_snapdistill_h1024_30k_1xh100` with
wandb_project pinned as a named verify delta (`d9dd385`) so
teacher-verbatim can never silently re-inherit a mainline write
target. Gate (a) identity oracle 6/6 bit-exact ✓; gate (b) drift eval
scoring at session end (train starts on pass; ~12–20 h wall to 30k +
endpoint evals). Discord: no inbound ×3 polls; pre-reg + re-bank
headlines posted. Lit slice ~10 min taken (ThinkProprio 2602.06575,
Cloak 2606.22836 → banked into #9/#11). Queue: local → SnapFlow
babysit (in-run s=t divergence watch; 10k record-only probe when a
quiet GPU appears; endpoint reads ~tomorrow) → draws-fairness ES
column + panel-v2 + stage-2b + E4B follow-on awaiting owner steer;
box → arm C babysit → 40k boundary ~12:2xZ → panel + masked reads
~15:3xZ (results instrument prep = next CPU work item, oracle before
data per the box-batch pattern); ≥2 ✓. GPUs busy (SnapFlow local, arm
C box) + CPU queue non-empty (arm-C results instrument, #16 follow-ups)
→ `run_work_next` armed per no-idle-pauses.*

*Previous update 2026-08-06 07:48–07:5xZ (real `date -u`) — tick: **flip re-bank
healthy and running HOT — @1,792/25,800 frames at 07:50:09Z, measured
480 f/min over a 60-s window (fastest this panel has run; prior evals
130–280 f/min), util bursty (99–100% bursts / 0% inter-batch gaps) but
throughput is the decider → scoring lands ~08:3x–08:4xZ, AHEAD of the
launch-time ~09:2x–09:3xZ estimate.** Reads at the boundary: adopt the
stable-key anchor iff chunk_mae inside band [6.4882, 6.7582] AND
state-copy/AR control rows bitwise-match the banked npz. Discord: no
new messages, no new reactions (history-checked; last 5 are our own
posts, latest the 07:45Z fairness-results headline). Queue unchanged:
local → flip re-bank reads (~08:4xZ, pulled in ~45 min) → SnapFlow
distill (launch-ready, needs the GPU the flip frees) + #9
state-dropout pre-reg (CPU, sanctioned); box → idle awaiting owner
steer (E4B follow-on + panel-v2 3 decisions + stage-2b + ES-column
adoption); ≥2 ✓. GPU busy (flip re-bank) + CPU queue non-empty (#9
pre-reg, re-bank boundary now <1 h out) → `run_work_next` armed per
no-idle-pauses; the chained work session takes the #9 pre-reg and the
re-bank boundary.*

*Previous update 2026-08-06 07:20–07:5xZ (real `date -u`) — work session: **FAIRNESS READS
IN — THE UNFAIR-PENALTY SIGNATURE FIRED ON ALL FOUR PRE-REGISTERED
READS: chunk MAE is measurably punishing flow for committing to
modes** ([results](posts/2026-08-06-draws-fairness-results.md),
report `reports/analysis__draws_fairness_k4l2.json`). Probe completed
07:39Z (2,458/2,458, zero interventions after the 06:57Z relaunch);
E1 gate passed (draw-0 re-decode drift 0.0145 < 0.05; degenerate
draws=1 oracle re-run green before the npz was opened). The reads:
(1) mean-of-10 5.4113/1.3975 on the probe subset — cross-checks the
chain's full-panel 5.365; (2) **best-of-10 3.8597, 2.01 BELOW AR's
paired 5.8680** — forgive mode choice and flow wins big; (3) paired
deficit monotone across dispersion quartiles **0.23→0.60→0.87→1.42**
(q4 = 6.2× q1, Spearman +0.13); (4) **energy score: flow 5.9308 vs
AR 8.7696** — flow wins the strictly proper score while losing
single-draw MAE on the same frames. Honest residual stated: deficit
positive even in the tight quartile (+0.23), win rate < 0.5
everywhere — partly artifact, NOT wholly; ES is now the candidate
distributional column for comm-holdout flow rankings (adoption =
owner decision, posted). **σ_draw direct = 0.02367 — SUPERSEDES the
0.0159 model pin (1.49×) but both floors hold (`reopen_floors:
false`) → re-bank band [6.4882, 6.7582] and SnapFlow adopt ≤ 6.7732
both numerically UNCHANGED** (`sigma_draw_direct.py` self-oracles
green; pooled-level cross-estimator 0.02522 inside the χ²₉ band).
Then, per the mantra (GPU freed 07:39Z): **#18.2 STABLE-KEY FLIP
RE-BANK LAUNCHED 07:41Z** (tmux `stablekeyrebank`,
`~/eval_flow80k_stablekey_rebank.sh` — NEW launcher with the σ_draw
gate asserted in-launcher; quiet-GPU guard passed; first-poll rule:
scoring @32/25,800 at 07:46:46Z, **99% util** → full-panel reads
~09:2x–09:3xZ: band + bitwise state-copy/AR controls). `check.py`
235 green; ideas #1/#12/#18.2 updated; blog built + Space pushed
(post URL 200); Discord posted 07:4xZ (no inbound traffic ×2 polls).
Queue: local → flip re-bank reads (~09:2x–09:3xZ: adopt iff inside
band AND controls bitwise) → SnapFlow distill (launch-ready, needs
the GPU the flip frees) + #9 state-dropout pre-reg (CPU,
sanctioned); box → idle awaiting owner steer (E4B follow-on +
panel-v2 3 decisions + stage-2b + now ES-column adoption); ≥2 ✓.
GPU busy (flip re-bank) + CPU queue non-empty (#9 pre-reg,
re-bank reads ~1.7 h out) → `run_work_next` armed per
no-idle-pauses; the chained work session takes the #9 pre-reg in
the babysit window.*

*Previous update 2026-08-06 07:18–07:2xZ (real `date -u`) — tick: **relaunched fairness
probe healthy past halfway** — @1,312/2,458 frames at 07:19:26Z, **100%
util**, log fresh (~60–90 f/min through this stretch; the pre-crash run
took ~28 min wall for the same 2,458 frames) → scoring done ~07:3xZ,
chained in-launcher CPU fairness reads open right after (E1 gate: draw-0
re-decode of the banked single-draw, drift < 0.05; direct σ_draw vs the
0.0159 pin). Discord: no new messages, no new reactions
(history-checked; last two are our own 06:16Z state-probe results +
06:58Z crash-fix posts). Pre-staged SUMMARY.md line for the
draws-fairness results skeleton committed with this tick (the skeleton
itself landed in da9ec6a; mdbook drops unlisted files — line belongs
with it). Queue unchanged: local → fairness reads (~07:3xZ, chained
in-launcher) → noise-key flip re-bank (band final) → SnapFlow distill +
#9 state-dropout pre-reg (CPU); box → idle awaiting owner steer (E4B
follow-on + panel-v2 3 decisions + stage-2b); ≥2 ✓. GPU busy (probe
endgame) + CPU queue non-empty (fairness reads ~15 min out,
state-dropout pre-reg) → `run_work_next` re-armed (driver had consumed
the 06:5xZ marker) per no-idle-pauses; the chained work session takes
the probe boundary + reads.*

*Previous update 2026-08-06 06:52–07:0xZ (real `date -u`) — tick: **FAIRNESS PROBE
CRASHED AT THE MERGE — THE MIRROR OF THE 04:4xZ BUG — FIXED,
RELAUNCHED 06:57Z (reads slip ~06:5x → ~07:3xZ).** All 2,458 frames
scored, then `merge_shards` IndexError from the OPPOSITE direction of
a433db9: this was the first --dump-draws-WITHOUT---dump-predictions
eval through the merge path — `cli.py` pre-creates empty per-policy
dump_predictions lists even when the flag is off, and the 2,458-row
dump_index permutation was applied to them; a433db9 guarded only the
dump_draws side. Fix `da9ec6a`: same empty-means-off guard at the
call site + mirror regression test — `check.py` 235 green (the
commit also lands the PREVIOUS session's pre-staged reads tooling:
draws-fairness results-post skeleton, `sigma_draw_direct.py` + its
tests). Crash log preserved (`.crashed-0652Z`); no banked numbers
touched (crash was post-scoring, pre-write). Probe relaunched 06:57Z
(same launcher, quiet-GPU guard passed, tmux `fairnessprobe`);
first-poll rule CONFIRMED in-session: scoring @32/2,458 at 07:03:30Z,
**99% util** — probe done ~07:1x–07:2xZ, chained reads ~07:2x–07:3xZ. Discord: no new messages, no new reactions (history-checked);
crash+fix+slip posted 07:0xZ. Queue unchanged: local → fairness
reads (now ~07:3xZ, chained in-launcher: E1 draw-0 re-decode gate
drift < 0.05 + direct σ_draw vs the 0.0159 pin) → noise-key flip
re-bank (band final) → SnapFlow distill + #9 state-dropout pre-reg
(CPU); box → idle awaiting owner steer (E4B follow-on + panel-v2 3
decisions + stage-2b); ≥2 ✓. GPU busy (probe re-run) + CPU queue
non-empty (fairness reads ~30 min out, state-dropout pre-reg) →
`run_work_next` armed per no-idle-pauses; the chained work session
takes the probe boundary + reads.*

*Previous update 2026-08-06 06:30–06:3xZ (real `date -u`) — tick: **fairness probe healthy
on its first tick-poll** — @672/2,458 frames at 06:29:50Z (log fresh,
99% util confirmed at launch + still 99% this poll), ~160 f/min →
probe lands ~06:4xZ, chained in-launcher CPU fairness reads
~06:4x–06:5xZ (E1 gate: draw-0 must re-decode the banked single-draw,
drift < 0.05; direct σ_draw measurement cross-checks the 0.0159 pin
before the SnapFlow/re-bank bands are consumed). Discord: no new
messages, no new reactions (history-checked; last 5 are our own
posts, latest the 06:16Z probe-results headline). Queue unchanged:
local → fairness reads (~06:5xZ) → noise-key flip re-bank (band
final) → SnapFlow distill (needs the GPU the probe frees) + #9
state-dropout pre-reg (CPU, sanctioned); box → idle awaiting owner
steer (E4B follow-on + panel-v2 3 decisions + stage-2b); ≥2 ✓. GPU
busy (probe) + CPU queue non-empty (reads ~15 min out, state-dropout
pre-reg) → `run_work_next` armed per no-idle-pauses; the chained
work session takes the probe boundary + reads.*

*Previous update 2026-08-06 06:03–06:3xZ (real `date -u`) — work session: **STATE-RELIANCE
PROBE READ — SUPPORTED: aux-off leans harder on the state shortcut,
D = Δ_first(B) − Δ_first(A-s0) = +0.702, CI95 [0.498, 0.916] — 14×
the pre-registered 0.05 threshold**
([results](posts/2026-08-06-state-probe-results.md), instrument
`fontaine/scripts/state_probe_results.py`, report
`reports/analysis__state_probe_q4.json`). Arm 4 (B masked) landed
06:06Z; the reads instrument was built + 3-way oracled in the ~15-min
window before it (degenerate all-zero/CI[0,0]; synthetic 1.10×
inflation → known-magnitude D detected AND common-effect cancellation
proven; misaligned-index abort — first oracle draft's "row shuffle"
was itself caught as NOT a pairing break, since pairing is by index).
All pre-registered execution oracles green at read time: state-copy/
-norm byte-match banked on all 4 arms (pairing + mask isolation
bitwise), truth/valid byte-identical, mask_state recorded, plan
sha256 asserted. Secondary chunk read agrees (+0.389 [0.106,
0.674]); all three banked expectations came true (Δ_chunk +15.3–16.4
every arm — absolute Δs stay descriptive per the stated OOD
limitation; no masked arm beats intact state-copy first; D > 0).
**Story now coherent with box-batch: B's better intact first_mae
(3.43 vs 3.87 subset) is bought with heavier state reliance — aux
supervision shifts representation toward vision. Branch rule fired:
#9 state-DROPOUT promoted, owed its own pre-reg** (ideas #9/#11
updated; GAP progress-conditioned Δ_first cut noted as discussion
material). `check.py` 229 green; blog built + Space pushed (post URL
200; post added to SUMMARY.md — mdbook silently drops unlisted
files, caught at the 404); Discord posted 06:2xZ. Then, per the
mantra (GPU went idle 06:06Z): **FAIRNESS PROBE LAUNCHED 06:24Z**
(tmux `fairnessprobe`, `~/eval_flow80k_drawsprobe_dump.sh`, quiet-GPU
guard passed; draws=10 heun-30 stride-7 2,458 frames + `--dump-draws`;
first-poll rule: 99% util, ~160 f/min → probe + chained CPU fairness
reads land ~06:4x–06:5xZ; E1 gate: draw-0 must re-decode banked
single-draw, drift < 0.05; its direct σ_draw measurement is the
pre-declared cross-check on the 0.0159 pin). Discord: no inbound.
Queue: local → fairness reads (~06:5xZ, chained in-launcher) →
noise-key flip re-bank (band final) → SnapFlow distill (launch-ready,
needs the GPU the fairness probe frees); **NEW queue-refill item: #9
state-dropout pre-reg (CPU, sanctioned by the fired branch rule)**;
box → idle awaiting owner steer (E4B follow-on + panel-v2 3 decisions
+ stage-2b); ≥2 ✓. GPU busy (fairness probe) + CPU queue non-empty
(fairness reads ~20 min out, state-dropout pre-reg) →
`run_work_next` armed per no-idle-pauses.*

*Previous update 2026-08-06 05:42–06:0xZ (real `date -u`) — work session: **σ_DRAW
FINALIZED = 0.0159 — BOTH PRE-REGISTERED FLOORS BIND; the SnapFlow
launch's last CPU-side blocker is closed**
([amendment](posts/2026-08-06-sigma-draw-finalization.md), instrument
`fontaine/scripts/sigma_draw_finalize.py`, report
`reports/analysis__sigma_draw_finalization.json`). The draws chain
dumped pooled JSONs only (no per-draw npz), so the pin is model-based
from the mean-of-N curve at matched solver: element error `bias + s·η`
(draw noise rank-1 within frame — worst case for pooled variance),
calibrated on (N=1, N=10); the gaussian-bias family predicts the
**held-out N=5 point to 0.087%** (stress families rejected at 2%/46%;
fitted systematic asymptote √c ≈ 5.21° nearly solver-independent).
σ_draw = std_η(frame-MAE)/√F_eff (F_eff = 16,488.5 valid-weighted):
heun-30 0.0140, heun-10 **0.0159** (pin = max; 1-NFE endpoint leans
low-step). **Bands now numeric before any dependent data: SnapFlow
adopt iff 1-NFE chunk_mae ≤ 6.7732 (3σ = 0.048 < 0.15 floor);
stable-noise re-bank band [6.4882, 6.7582] (σ < 0.045 floor).**
Verdict family-independent — even the a-priori-max pure-noise reading
(0.040) stays under both floors; the fairness probe's `--dump-draws`
direct measurement supersedes if larger (lands before either
dependent eval opens). Amendment blockquotes added to both amended
pre-regs; ideas #12/#18.2 updated. Oracles per charter: MC end-to-end
on the calibrated family (m(N) <0.5%, pooled σ <15%), LS recovery to
1e-10, flat/inverted clamp, posted-number input asserts + 7 new tests
— **`check.py` 229 green.** Blog built + Space pushed (post URL 200);
Discord posted 06:00Z. Probe babysits 05:42/05:51/05:58Z: **arm 3
(A-s0 masked) COMPLETE 05:49Z** (npz+JSON+HTML), arm 4 (B masked, the
last) @1,632/4,301 at 05:58Z, ~250 f/min → **all four state-probe
reads open ~06:1xZ**. Discord traffic: none inbound. Queue: local →
state-probe reads (~06:1xZ, instrument armed) → fairness probe →
noise-key flip re-bank (band now final) → SnapFlow distill (launch
path fully unblocked, needs quiet GPU); box → idle awaiting owner
steer (E4B follow-on + panel-v2 3 decisions + stage-2b); ≥2 ✓. GPU
busy (probe arm 4) + CPU queue non-empty (probe reads ~15 min out) →
`run_work_next` armed per no-idle-pauses.*

*Previous update 2026-08-06 05:30–05:4xZ (real `date -u`) — work session: **E4B
NO-LAUNCH — THE PRE-REGISTERED TERMINAL BRANCH FIRED: all four memory-
ladder rungs OOM'd on 80 GB; the screen does not launch under the
matched recipe** ([finding post](posts/2026-08-06-e4b-no-launch.md),
Amendment 2 finalized in the
[pre-reg](posts/2026-08-05-prereg-e4b-screen.md)). The chained
session's smoke had already run rungs 2–4 (2×6 / 3×4 / 4×3 chunked
backward, correct banners, box code 9ddcfe3): peaks 81,035–81,059 of
81,559 MiB, every rung dead in the FIRST train_step — **zero
optimizer steps ⇒ Adam fp32 m/v (~31.8 GiB for 3,975.3M live params)
never allocated, so the true need is ~≥110 GiB/rank, not a near-miss**
(consistency: E2B ran 71–75 GiB at ~2.2× fewer live params). Read:
feasibility negative, NOT a scale answer — probe/panel gates never
ran, attribution question stays open; E4B's zero-port-cost premise is
dead, so Molmo2-4B (survey rank 2) competes near-even with any
ZeRO-1 re-entry (which would be a NEW pre-reg). Follow-on decision
posted to the owner with 4 options + recommendation (drop E4B →
box to #11 grounding arms after the probe read; ZeRO-1 re-entry
queued as a candidate). Blog built + Space pushed (post URL 200;
link-fix follow-up posted after a wrong hostname in the first
Discord message). `check.py` 222 green. Probe babysits 05:30/05:38Z:
arm 3 (A-s0 masked) @1,152→2,432/4,301, 83% util, ~160 f/min — arm 3
lands ~05:5xZ, arm 4 keeps reads on pace **~06:1x–06:4xZ**. Discord:
no new messages. Queue: local → state-probe reads (~06:1x–06:4xZ) →
σ_draw amendment (CPU-ready: draws runs 3–5 all in) + fairness probe
→ SnapFlow distill; box → idle awaiting owner steer (E4B follow-on
NEW + panel-v2 3 decisions + stage-2b); ≥2 ✓. GPU busy (probe arms
3–4) + CPU queue non-empty (σ_draw amendment, probe reads) →
`run_work_next` armed per no-idle-pauses.*

*Previous update 2026-08-06 05:29–05:3xZ (real `date -u`) — tick: **state probe
past halfway — arms 1+2 COMPLETE with full npz+JSON+HTML (the a433db9
merge fix held), arm 3 (A-s0 masked) scoring.** AR-100k masked and
flow-80k masked landed 05:04Z / 05:19Z; arm 3 @832/4,301 at 05:28:51Z,
log fresh, ~100–160 f/min through load — arms 3+4 keep the reads on
pace ~06:1x–06:4xZ. No surface reads taken (frozen paired reads run
via the instrument once all four arms land). Discord: no new messages,
no new reactions (history-checked; last message is our own 04:43Z
crash+fix post); panel-v2 (3 decisions) + stage-2b still await owner
steer. E4B finalization amendment draft still staged in-tree — the
chained work session's items: B12 memory smoke on the idle box →
amendment finalize → E4B launch. Queue unchanged: box → B12 smoke →
E4B amendment → E4B launch; local → state-probe reads (~06:1x–06:4xZ)
→ σ_draw amendment + fairness probe → SnapFlow distill; +panel-v2 +
stage-2b awaiting steer — ≥2 ✓. GPU busy (probe arm 3) + CPU queue
non-empty (E4B items) → `run_work_next` armed per no-idle-pauses.*

*Previous update 2026-08-06 04:56–04:5xZ (real `date -u`) — tick: **relaunched
state probe healthy and warmer than run 1** — arm 1 (AR-100k masked)
@2,432/4,301 at 04:56:44Z, ~262 frames/min sustained since the 04:51
poll (95% util confirmed then), log fresh; arm 1 lands ~05:04Z, all
four reads on pace ~06:1x–06:4xZ. Discord: no new messages, no new
reactions (history-checked; last message is our own 04:43Z crash+fix
post); panel-v2 (3 decisions) + stage-2b still await owner steer.
E4B finalization amendment draft remains staged in-tree
(PENDING_PEAK/PENDING_RUNG await the B12 smoke) — the chained work
session's items: B12 memory smoke on the idle box → amendment
finalize → E4B launch. Queue unchanged: box → B12 smoke → E4B
amendment → E4B launch; local → state-probe reads (~06:4xZ) → σ_draw
amendment + fairness probe → SnapFlow distill; +panel-v2 + stage-2b
awaiting steer — ≥2 ✓. GPU busy (probe) + CPU queue non-empty (E4B
items) → `run_work_next` armed per no-idle-pauses.*

*Previous update 2026-08-06 04:39–04:5xZ (real `date -u`) — tick: **STATE PROBE
CRASHED AT THE MERGE — DIAGNOSED, FIXED, RELAUNCHED (reads slip ~06Z →
~06:4x–07:0xZ).** Arm 1 (AR-100k masked) scored ALL 4,301 frames then
died in `merge_shards`: `permuted()` applied the dump_index row order
to `dump_draws`, which is `[]` when `--dump-predictions` runs without
`--dump-draws` — IndexError before any npz/JSON/report was written,
and the launcher's `set -euo pipefail` took the whole 4-arm chain +
tmux session down with it. Mechanism owned: the line landed in
f0868b3 (fairness instrument added dump_draws to the shard merge);
every local eval since had passed `--dump-draws` (draws runs 3–5), so
the probe's AR arm was the FIRST dump-predictions-without-dump-draws
eval through the new path — the test fixture always filled
dump_draws, which is why 221 stayed green. Fix a433db9: empty-means-
off guard at the call site + regression test (`check.py` 222 green).
Probe relaunched 04:44Z (same launcher, sha256 re-asserted, tmux
`stateprobe`); first-poll util rule CONFIRMED post-load: arm 1
@1,152/4,301 at 04:51:51Z, **95% util**, ~280 frames/min (warmer than
the first run's ~160) — reads land ~06:1x–06:4xZ. Also found in-tree: the
E4B finalization amendment DRAFT (Amendment 2) already staged in the
pre-reg post — σ_seed 0.038 section complete, PENDING_PEAK/
PENDING_RUNG placeholders await the B12 memory smoke on the idle box
→ that smoke + amendment + E4B launch are the chained work session's
items. Discord: no new traffic (history: our 04:24Z results post,
no new reactions); crash+fix+slip noted in-channel; panel-v2 (3
decisions) + stage-2b still await owner steer. Queue: box → B12
smoke → E4B amendment finalize → E4B launch; local → state-probe
reads (~07Z) → σ_draw amendment + fairness probe → SnapFlow distill;
+panel-v2 + stage-2b awaiting steer — ≥2 ✓. GPU busy (probe re-run)
+ CPU queue non-empty (E4B items) → `run_work_next` armed per
no-idle-pauses.*

*Previous update 2026-08-06 04:28–04:3xZ (real `date -u`) — tick: **state-reliance
probe healthy on its first tick-poll** — arm 1 (AR-100k masked) @2,112/4,301
frames, log fresh 04:29Z, util 75% (first-poll rule re-confirmed; ~160
frames/min, on pace), all four reads still land ~06:0xZ; policy name
carries `_state-masked` as registered. Box: all 4 GPUs idle by design
(post-batch); E4B finalization amendment (σ_seed 0.038 in hand,
CPU-side) + B12 memory smoke remain the next box items → E4B launch.
Discord: no new traffic (the one unread message was our own 04:24Z
results headline; history-checked, no new reactions); panel-v2 (3
decisions) + stage-2b still await owner steer. Queue unchanged: box →
E4B amendment + smoke → E4B launch; local → state-probe reads (~06Z)
→ σ_draw amendment + fairness probe → SnapFlow distill; +panel-v2 +
stage-2b awaiting steer — ≥2 ✓. GPU busy (probe) + CPU queue
non-empty (E4B amendment) → `run_work_next` armed per
no-idle-pauses.*

*Previous update 2026-08-06 03:51–04:3xZ (real `date -u`) — work session: **BOX-BATCH
RESULTS ARE IN AND THE DECISION RULE FIRED: THE AUX-OFF EFFECT IS REAL
— aux supervision helps action prediction, the mainline "within noise"
expectation is falsified**
([results post](posts/2026-08-06-box-batch-results.md),
`reports/analysis__box_batch_40k_k4l2.json`). s1/s2 landed 04:0x–04:1xZ
(watched through the boundary), all four npz+JSON pairs pulled, the
pre-built instrument ran the frozen reads: arms A-s0/s1/s2
**7.7966/7.8052/7.7355**, B (aux-off) **8.2989**; primary paired
per-frame B−A-s0 **+0.462, CI95 [0.387, 0.537]** — 7.5× the largest
replicate delta (0.0697, within the ≤0.2 soft expectation),
leave-one-repo-out coherent (worst exclusion +0.435). **σ_seed(chunk)
= 0.038 → E4B adopt band = max(3σ, 0.15) = 0.15 (floor binds).** The
twist survived pooling: B's first_mae 3.5009 BEATS aux-on (3.94–4.11),
cond-sensitivity 1.13 vs 1.86–2.00, predictions 8% closer to
state-copy — the state-shortcut story is coherent but stays
descriptive until the probe's frozen reads. Ledger's first training
rows added; ideas #6 → confirmed (aux stays ON in all future recipes).
Also this session: stranded parity-extension work found in the tree
(prior session hit its cap before committing) — `check.py` 221 green,
committed `70bda9a`. Local: **draws run 5 (draws=1 heun-10) COMPLETED
04:12Z: 6.8468/2.3525** (vs heun-30 draws-1 6.6232/1.9331: heun-10
costs +0.22/+0.42 at single draw) — **the draws chain (runs 1–5) is
COMPLETE**; σ_draw amendment + fairness probe are the next local
items. Then, per the mantra (both boxes went idle): **STATE-RELIANCE
PROBE LAUNCHED 04:2xZ** on the freed local GPU (tmux `stateprobe`,
`~/launch_state_probe_q4.sh`, plan sha256 asserted at launch, 4
sequential masked runs ≈1.7 GPU-h; first-poll rule: arm 1 (AR-100k)
scoring at ~120–170 frames/min, util 70%, policy name carries
`_state-masked` as registered) — all four reads land ~06:0xZ. Box: all
4 GPUs idle; E4B finalization amendment (σ_seed now in hand, CPU-side)
+ B12 memory smoke are the next box items → E4B launch. Discord: no
new traffic; results headline posted; panel-v2 (3 decisions) +
stage-2b still await owner steer. Queue: box → E4B amendment + smoke →
E4B launch; local → state-probe reads (~06Z) → σ_draw amendment +
fairness probe → SnapFlow distill; +panel-v2 + stage-2b awaiting steer
— ≥2 ✓. GPU busy (probe) + CPU queue non-empty (E4B amendment) →
`run_work_next` armed per no-idle-pauses.*

*Previous update 2026-08-06 03:48–03:5xZ (real `date -u`) — tick: **A-s0'S PANEL
READ IS IN — chunk_mae 7.7966 / first_mae 3.9422 @40k** (report+npz+
HTML landed on the box 03:40Z, pulled local this tick by direct
rsync; state-copy reproduces 11.7848/2.6202). The results
instrument's primary read (B−A-s0) and the state-reliance probe are
BOTH unblocked on inputs — first surface read: B (aux-off) 8.2989 is
+0.50 WORSE on chunk_mae (the pre-registered primary's direction),
but B's first_mae 3.5009 is BETTER than A-s0's 3.9422, and BOTH arms
sit above state-copy 2.6202 on first_mae — the "B's flag = aux-off
state-shortcut" story just got complicated before the probe even
runs; numbers stay surface-only until `box_batch_results.py` does
the paired reads on all four npzs. s1 @22.9k, s2 @23.9k of 25.8k
(logs fresh 03:49Z, ~160–190 frames/min) → both land ~04:0x–04:1xZ;
GPU0/GPU1 now idle (A-s0 done + E4B slot). Local: **draws run 4
(draws=10 heun-10) COMPLETED 03:38Z: 5.4045/1.5319** vs run 2's
heun-30 5.365/1.424 — the solver-step cost at draws=10 is small
(+0.04 chunk); run 5 (draws=1 heun-10) chained and scoring @100%
util. Discord: no new messages, no new reactions (history-checked);
panel-v2 (3 decisions) + stage-2b still await owner steer. Queue:
box → **results window OPEN once s1/s2 land ~04:1xZ** (instrument
armed; then σ_seed → E4B amendment + smoke on the freed GPUs) +
state-reliance probe (4 masked runs, first quiet window after the
evals clear); local → run 5 → fairness probe → SnapFlow distill;
+panel-v2 + stage-2b awaiting steer — ≥2 ✓. GPUs busy + the results
read is ~15–25 min out → `run_work_next` armed; the chained work
session takes the full results window.*

*Previous update 2026-08-06 03:04–03:0xZ (real `date -u`) — tick: **all chains
healthy; box lead arm A-s0 is ~40 min from its read — the ~04Z
results window opens next session.** Box: three control evals
scoring — A-s0 @19.1k, s1 @14.1k, s2 @15.2k of 25.8k, all advancing
since the 02:58 poll (~130–210 frames/min), logs fresh 03:03–03:04Z;
A-s0 lands ~03:4x–03:5xZ (its npz unblocks BOTH the results
instrument's primary read and the state-reliance probe), s1/s2
~04:1x–04:2xZ; B complete (pulled 02:09Z); GPU1 idle as decided.
Local draws run 4 @17.2k/25.8k, 99% util, log fresh 03:04Z, on
pacing ~04:1xZ. Discord: no new messages; history-checked — no new
reactions beyond the recorded ❤️; panel-v2 (3 decisions) + stage-2b
still await owner steer. Queue unchanged: box → results post (~04Z,
instrument armed) + E4B smoke/σ_seed/amendment → E4B launch; local →
fairness probe → SnapFlow distill; + state-reliance probe (unblocks
on A-s0's npz ~03:5xZ, slots any quiet GPU window); +panel-v2 +
stage-2b awaiting steer — ≥2 ✓. GPUs busy + CPU queue non-empty →
`run_work_next` armed per no-idle-pauses; the chained session takes
the A-s0 boundary.*

*Previous update 2026-08-06 02:49–03:1xZ (real `date -u`) — work session: **STATE-RELIANCE
PROBE PRE-REGISTERED (#11 rung (a)) — the lit slice's state-dominant-
bias mechanism now has its falsification instrument landed and its
reads frozen, one session after the mechanism was named**
([pre-reg](posts/2026-08-06-prereg-state-reliance-probe.md)).
Instrument: `bijou.eval --mask-state` substitutes each item's
per-dataset state MEAN, so the normalized soft state token collates
to EXACTLY zero (x−x ≡ 0 bitwise) — zero state information at
in-distribution magnitude; applied in `apply_overrides` so the
narrated pass sees identical inputs; policy name gains
`_state-masked` (the `_drawsN` can't-pass-as-deployment precedent);
report JSON + npz scalars + banner all record it; parse guards
(no-checkpoint, --smolvla mix) die at the parser; baselines
deliberately intact — state-copy stays the reference AND becomes the
execution oracle (masked run's baseline rows must byte-match the
banked npz pooled on the subset rows: proves pairing + mask
isolation). 6 new tests (`tests/test_mask_state.py` exactly-zero /
at-mean identity / no-mutation + 3 parse guards), `check.py` 221
green. Design: frozen 4,301-row subset plan
(`plans/holdout_curated_v0_k4l2_stateprobe_q4.json`, every 4th core
row, sha256-pinned, builder+oracle in
`fontaine/scripts/state_probe_subset_plan.py`) — a strict row-subset,
so the intact side POOLS from banked npzs (AR-100k, flow-80k, B in
hand; A-s0 ~04Z): 4 masked runs ≈ 1.7 GPU-h total, zero intact
re-evals. Primary read frozen: **D = Δ_first(B) − Δ_first(A-s0)**,
paired seeded bootstrap; supported iff CI excludes 0 AND D ≥ 0.05 —
supported ⇒ #9 state-DROPOUT gets its own pre-reg; not ⇒ the
mechanism is dropped as B's-flag explanation. OOD limitation stated
honestly (masking is untrained; the paired B−A-s0 difference
subtracts the common OOD effect). Blocked on A-s0's ~04Z npz; first
quiet GPU window, never beside a pre-registered eval. Babysits
02:49/02:58Z: box three control evals scoring — A-s0 @17.8k, s1
@13.2k, s2 @14.0k of 25.8k (~160–210 frames/min), reads on pace
~03:4x–04:1xZ; B complete (pulled); GPU1 idle as decided. Local
draws run 4 @15.4k/25.8k, 99% util, on pacing ~04:1xZ. Discord: no
new messages; pre-reg posted 03:0xZ; panel-v2 (3 decisions) +
stage-2b still await owner steer. Queue: box → results post (~04Z,
instrument armed) + E4B smoke/σ_seed/amendment → E4B launch; local →
fairness probe → SnapFlow distill; + state-reliance probe (NEW,
blocked on ~04Z npz, slots any quiet GPU window); +panel-v2 +
stage-2b awaiting steer — ≥2 ✓. Blog built + Space pushed. GPUs busy
+ CPU queue non-empty → `run_work_next` armed per no-idle-pauses.*

*Previous update 2026-08-06 02:48–02:5xZ (real `date -u`) — tick: **all chains
healthy; box endgame past 60% on the lead arm.** Box: three control
evals scoring — A-s0 @15.9k, s1 @11.7k, s2 @12.4k of 25.8k, all
advancing since the 02:39 poll (~130–160 frames/min), logs fresh
02:47–02:48Z, reads on pace ~03:4x–04:1xZ; B complete (pulled
02:09Z); GPU1 idle as decided (E4B smoke at the boundary). Local
draws run 4 @12.8k/25.8k, 91% util, log fresh 02:48Z, on pacing
~04:1xZ. Discord: no new messages; history-checked — no new
reactions beyond the recorded 👍/❤️; panel-v2 (3 decisions) +
stage-2b still await owner steer. Queue unchanged: box → results
post (~04Z, instrument armed) + E4B smoke/σ_seed/amendment → E4B
launch; local → fairness probe → SnapFlow distill; +panel-v2 +
stage-2b awaiting steer — ≥2 ✓. GPUs busy + CPU queue non-empty →
`run_work_next` armed per no-idle-pauses.*

*Previous update 2026-08-06 02:39–02:5xZ (real `date -u`) — work session: **THE
OVERDUE LIT SLICE TAKEN (the flagged MUST — 7 sessions deferred) —
and it named a candidate mechanism for the batch's biggest open
flag: state-dominant bias.** Targeted pass on distillation, data
curation, and grounding. Bankings (ideas #11 + #9): (1)
**[ReViP](https://arxiv.org/abs/2601.16667)** diagnoses VLA "false
completion" as modality imbalance — policies over-rely on
proprioceptive state progression and under-use vision (+26% over π0
via progress-aware FiLM rebalance; abstract-depth). The causal-
confusion line ([2506.23944](https://arxiv.org/abs/2506.23944)
p=0.8 state masking; [2509.18644](https://arxiv.org/abs/2509.18644)
state-FREE policy) agrees: proprioception is the shortcut, vision
generalizes. Direct hit on our grounding gap AND on **B's pending
first_mae 3.5009 > copy 2.6202 flag** — aux-off leaning harder on
the state shortcut is now a named, testable hypothesis. (2)
Cheapest falsification queued in #11: a **state-reliance probe**
(panel-subset eval with state zeroed vs intact on AR-100k/flow-80k/
B/A-s0; Δ(B) vs Δ(A-s0) is the read; needs `--mask-state` +
its own pre-reg). (3) #9's "state-noise" sharpened to **state
DROPOUT** (the literature's lever) as the paired train-time arm.
Skim-depth: [2602.09722](https://arxiv.org/abs/2602.09722)
(heterogeneous pooling → negative transfer; supports selective
mixture) and the [data-engine survey](https://arxiv.org/abs/2604.23001)
(dedup/contamination = THE bottleneck — the #18.7 census is exactly
this). No new launch implied; the probe idea feeds the ~04Z results
read's discussion, not its frozen numbers. `check.py` 215 green.
Babysits 02:39/02:42Z: box controls A-s0 @14.6k, s1 @10.9k, s2
@11.4k of 25.8k — advancing, reads ~03:4x–04:1xZ; B complete
(pulled); GPU1 idle as decided. Local draws run 4 @11.4k/25.8k, log
fresh, on pacing ~04:0x–04:1xZ. Discord: no new messages. Queue
unchanged: box → results post (~04Z, instrument armed) + E4B
smoke/σ_seed/amendment → E4B launch; local → fairness probe →
SnapFlow distill; +panel-v2 + stage-2b awaiting steer — ≥2 ✓. GPUs
busy + CPU queue non-empty → `run_work_next` armed per
no-idle-pauses.*

*Previous update 2026-08-06 02:37–02:4xZ (real `date -u`) — tick: **all chains
healthy; box endgame past the halfway mark.** Box: three control
evals scoring — A-s0 @13.6k, s1 @10.1k, s2 @10.8k of 25.8k, all
advancing since the 02:3x poll, reads on pace ~03:4x–04:1xZ; B
complete (pulled 02:09Z); GPU1 idle as decided (smoke at the
boundary). Local draws run 4 @10.1k/25.8k, log fresh 02:37Z, on
pacing ~04:0x–04:1xZ. Discord: no new messages; history-checked —
no new reactions beyond the recorded 👍/❤️; panel-v2 (3 decisions)
+ stage-2b still await owner steer. Queue unchanged: box → results
post (~04Z, instrument armed) + E4B smoke/σ_seed/amendment → E4B
launch; local → fairness probe → SnapFlow distill; +panel-v2 +
stage-2b awaiting steer — ≥2 ✓. **Lit slice remains a MUST for the
next work session** (7 sessions since 00:14Z). GPUs busy + CPU
queue non-empty → `run_work_next` armed per no-idle-pauses.*

*Previous update 2026-08-06 02:24–02:4xZ (real `date -u`) — work session: **Q3 TRIPWIRE
NOISE FIX LANDED (#18.3, deep-dive finding 3) — the conditioning-collapse
alarm now measures conditioning, not sampling variance, closed before
the SnapFlow distill launch (the next conditioned flow run,
`--condition-fields subgoal outcome smoothness`).** The tripwire's
override decode re-used the *advanced* generator — fresh noise — so
for a flow decoder mean|Δ| vs the scalar pass had a floor at the
sampling variance even for a fully conditioning-blind model, the
exact state the alarm was registered to catch. Fix:
`FlowDecoder.predict_chunk` now returns the noise it integrated
(`BijouPrediction.noise`; fallback draw moved from `sample_actions`
into `predict_chunk` — the identical randn), `validate()` captures it
per rich row, and the Q3 override decode reuses each flipped row's
scalar-pass noise, so |Δ| isolates the conditioning effect; AR path
byte-unchanged (noise None, greedy — was already exact). Oracles:
pre-edit banked reference on a seeded fixture reproduced bit-exact
post-edit (actions AND generator end-state — in-run probe curves
stay comparable across the change); noise round-trip bitwise;
eval/panel paths structurally untouched (eval always passes explicit
per-item noise — verified at both `policies.py` call sites). 3 new
tests (`tests/test_condition_tripwire.py`), `check.py` 215 green.
Semantics note recorded in ideas #18.3: flow-run
`condition_sensitivity` not comparable to mainline's historical
values (which carried the floor). Babysit 02:3xZ: box ×3 control
evals scoring A-s0 @13.2k, s1 @9.5k, s2 @10.1k of 25.8k — frames
advancing, on pace ~03:4x–04:1xZ; B complete (pulled 02:09Z); GPU1
idle as decided (smoke at the boundary). Local draws run 4 @9.5k/
25.8k, 99% util, on pacing ~04:0xZ. Discord: no new messages;
panel-v2 + stage-2b still await owner steer. Queue unchanged: box →
results post (~04Z, instrument armed) + E4B smoke/σ_seed/amendment →
E4B launch; local → fairness probe → SnapFlow distill (its
conditioned-run path now unblocked by this fix); +panel-v2 +
stage-2b awaiting steer — ≥2 ✓. Lit slice skipped again (6 sessions
since 00:14Z — #18.3 was the ladder's top unblocked item with a
launch-path deadline; the pure-babysit window before ~04Z or the
first post-results session takes the slice, stated as a MUST). GPUs
busy + CPU queue non-empty → `run_work_next` armed per
no-idle-pauses.*

*Previous update 2026-08-06 02:22–02:2xZ (real `date -u`) — tick: **all chains healthy;
box endgame is three control evals from done, and B's eval GPU is
now idle — E4B smoke deliberately deferred to the boundary.** Box:
B's panel eval COMPLETE (25,792/25,800 final scoring line; report+
npz+HTML confirmed pulled local 02:09Z by boxsync — the ~04Z
results run has its first input in hand), GPU1 idle as expected.
Controls scoring on GPUs 0/2/3: A-s0 @11.4k, s1 @7.1k, s2 @7.6k of
25.8k — on pace, reads ~03:4x–04:1xZ. **Judgment call recorded:
GPU1 free unblocks the E4B B12 memory smoke (`~/smoke_e4b_b12.sh`),
but it is NOT run this tick — a training smoke co-located beside
three live pre-registered evals risks the same CPU contention that
slowed the box 0.39→0.51 s/step during the parity job, and the
smoke is off the critical path (E4B launch waits on the σ_seed
finalization amendment, which needs the same ~04Z control reads).
Smoke runs at the eval boundary alongside the results work.** Local
draws run 4 (draws=10 heun-10) @6.0k/25.8k, ~190 frames/min, done
~04:0xZ, on pacing. Boxsync loop alive (marker discipline working).
Discord: no new messages; history-checked — no new reactions; the
panel-v2 amendment (3 decision points) still awaits owner steer.
Queue unchanged: box → results post (~04Z, instrument armed) + E4B
smoke/σ_seed/amendment → E4B launch; local → fairness probe →
SnapFlow distill; +panel-v2 + stage-2b awaiting steer — ≥2 ✓. GPUs
busy + CPU queue non-empty → `run_work_next` armed per
no-idle-pauses.*

*Previous update 2026-08-06 02:11–02:4xZ (real `date -u`) — work session: **PANEL-V2
AMENDMENT PROPOSED (#18.7 follow-on) — the panel re-definition is
frozen, its anchors derived, and the decision is on the owner's
desk before the ~04Z anchor boundary**
([amendment](posts/2026-08-06-panel-v2-amendment.md), instrument
`fontaine/scripts/panel_v2.py`, frozen plan
`plans/holdout_curated_v0_k4l2_panel_v2.json`, report
`~/panel_v2_anchors.json`). v2 = v1 minus the census's 524 leaked
episodes minus the 3 wrap-census corrupt repos (which the panel
still scored: 52 core rows averaging ~31° wrap-scale MAE) — a
strict row-subset (core 17,204→15,056, labeled 8,596→7,522, zero
overlap between the two exclusion sets), so **every banked npz
re-pools to v2 exactly, zero re-evals; adoption is CPU-only.**
v2 anchors, oracle-gated (v1 anchors + census clean-core both
reproduce exactly, state-copy pools identically from both npzs):
**AR-100k 5.8894/2.1396, flow-80k 6.7151/1.9453, state-copy
11.7639/2.5851** — the two exclusions partially offset (leak
removal +0.17–0.19, corrupt removal −0.09). Transition rules
proposed: in-flight pre-registered reads (box results ~04Z, draws
chain, E4B, SnapFlow) finish on v1 as registered with the v2 column
quoted alongside; v2 becomes the convention for NEW pre-regs on
approval; **the #18.2 noise-key flip (and optionally #14
shortest-arc) bundles at the same re-bank boundary so the flow
anchor re-banks once, not three times.** Three owner decision
points posted to Discord. `check.py` 212 green; synthetic
materialization oracle + all real-data asserts pass. Babysits
02:15/02:4xZ: box ×3 control evals scoring (A-s0 @9.3k, s1 @5.1k,
s2 @5.5k of 25.8k — reads on pace ~03:4x–04:1xZ), B's report+npz
landed; local draws run 4 @3.2k/25.8k, 100% util, on pacing.
Discord: no new messages. Queue unchanged: box → results post
(~04Z, instrument armed) → E4B; local → fairness probe → SnapFlow
distill; +panel-v2 amendment + stage-2b awaiting owner steer — ≥2
✓. GPUs busy + CPU queue non-empty → `run_work_next` armed per
no-idle-pauses.*

*Previous update 2026-08-06 02:07–02:2xZ (real `date -u`) — tick: **B'S PANEL READ IS IN —
the first of the four box-batch numbers: aux-off (B) chunk_mae
8.2989 / first_mae 3.5009 @40k** (state-copy 11.7848/2.6202;
report+npz landed 02:09Z, ahead of the ~02:15Z projection —
babysat through the boundary via a report-JSON watcher, scoring
confirmed complete 25,800/25,800 with the writer in CPU-side
post-processing before the files appeared). Flag for the results
read, stated now before the controls land: **B's first_mae 3.5009
is WORSE than the state-copy baseline 2.6202** — consistent with an
aux-off grounding cost, but no conclusion until the paired
per-frame reads; the pre-registered primary stays chunk_mae B−A-s0
via `box_batch_results.py` when all four npz+JSON pairs are in.
Controls on pace: A-s0 @8.5k, s1 @4.4k, s2 @4.7k of 25.8k — reads
~03:4x–04:1xZ. Boxsync loop alive (20-min passes; s1/s2
step_040000 marked synced_complete, B's report pulls on the next
pass). Local draws run 4 (draws=10 heun-10) @2.1k/25.8k, 99% util,
on pacing. Discord: no new messages; history check surfaced a
**❤️ on the draws-10 HTML report attachment post** — recorded as
lightweight positive ack (owner liked seeing the report). Queue
unchanged: box → results post (~04Z, instrument armed) → E4B;
local → fairness probe → SnapFlow distill; +panel-v2 amendment
awaiting owner steer — ≥2 ✓. GPUs busy + CPU queue non-empty →
`run_work_next` armed per no-idle-pauses.*

*Previous update 2026-08-06 01:39–02:1xZ (real `date -u`) — work session: **DUPLICATE-CONTENT
CENSUS EXECUTED (#18.7) — THE PANEL'S HOLDOUT IS BREACHED BY THE
FORK CHANNEL: 12.2% of core panel frames (2,096/17,204) score on
episodes with BYTE-EXACT twins in train**
([results post](posts/2026-08-06-dup-census-results.md),
`fontaine/scripts/dup_content_census.py` +
`dup_census_anchor_impact.py`, reports `~/dup_census_report.json` +
`~/dup_census_anchor_impact.json`). The deep-dive finding-7 census,
run deliberately BEFORE the box results land (~03–04Z read fine
holdout deltas at a 0.15 band): the corpus is heavily forked —
6,935/52,507 episodes (2.67M frames) in 3,348 cross-repo byte-exact
action+state clusters (quantized tier adds NOTHING — pure
re-uploads); **524 holdout episodes across 79 repos have train-side
twins** (same-user variants: samanthalhy herding, shylee pengrip,
dopaul chess; cross-user forks: Chojins↔bensprenger,
Dangvi↔s20000s, lirislab↔roboticshack). Anchor impact through a
join content-verified against raw parquet (partition reproduces
both anchors exactly): leaked frames score ~1.3–1.6 BETTER than
clean on both banked models — **clean-core anchors AR-100k
5.9761/2.1695, flow-80k 6.8137/1.9714** (published numbers
~0.17–0.19 optimistic in level; content-difficulty confound stated).
**Paired within-corpus deltas — box batch, E4B, draws chain — are
unaffected** (every arm shares the train corpus and the same leaked
frames); absolute generalization claims + the comm→rig bridge now
quote clean-core. Panel-v2 (excluding the 524) = a panel
re-definition → queued for its own amendment + owner steer;
exclusion list frozen. Validation: 7-case synthetic oracle; split
mirror PROVEN on all 878 plan repos (plan episodes == re-derived
holdout_episodes); 20-pair collision guard; zero structural
warnings corpus-wide. `check.py` 212 green. Babysits 01:39/01:51/
02:04Z: box ×4 eval chains all scoring — **B @24.5k/25.8k, its
panel read lands ~02:15Z**, A-s0 @7.7k, s1 @3.6k, s2 @3.9k (reads
~03:4x–04:1xZ); local **draws run 3 (draws=5) COMPLETED 01:53Z:
5.5235/1.4985** (monotone in N: 6.6232@1 → 5.5235@5 → 5.365@10),
run 4 (draws=10 heun-10, the solver-step arm) chained and scoring
@99% util. Discord: no new messages ×3 polls. Queue depth: box →
results post (~04Z, instrument armed) → E4B; local → fairness probe
→ SnapFlow distill; +panel-v2 amendment — ≥2 ✓. GPUs busy + CPU
queue non-empty → `run_work_next` armed per no-idle-pauses.*

*Previous update 2026-08-06 01:19–01:4xZ (real `date -u`) — work session: **BOX-BATCH
RESULTS INSTRUMENT LANDED + ORACLED BEFORE THE DATA — when the four
panel npz+JSON pairs land (~03–04Z), one command produces the
results-post numbers, the frozen decision verdict, AND both
finalization amendments** (`fontaine/scripts/box_batch_results.py`,
#6). Implements exactly the frozen reads of the box-batch pre-reg:
primary paired per-frame chunk_mae B−A-s0 with seeded bootstrap CI;
pairwise replicate deltas {s0,s1,s2}; E5 0.2/0.3 noise-floor bands;
the pre-registered decision rule (effect > LARGEST pairwise replicate
delta AND leave-one-repo-out coherent — sign + threshold must survive
every single-repo exclusion); σ_seed (ddof=1 over replicate pooled
chunk_maes) → E4B adopt band max(3σ_seed, 0.15) and rig-benchmark
slot 2. Headline column = bare `pred:bijou@STEP` by anchor
convention (the real eval JSONs keep per-policy summaries, so
"match the report" can't select — found by inspecting a live arm's
`--output-json` contract mid-build); each arm's report JSON is a
drift oracle instead: recomputed chunk+first must reproduce its
summaries entry (<5e-3) or abort. Four oracles all green: anchors
5.8026/2.1431 + 6.6232/1.9331 exact through this file's pooling;
degenerate same-npz → all-zero deltas, CI [0,0], band floor,
within-noise verdict; synthetic 1.05× error inflation → +0.27679
delta, real+coherent+correct sign (a flat +c prediction shift is
documented as an INVALID synthetic — balanced error signs cancel
the MAE shift; the first oracle draft made exactly that error and
the assert caught it); report cross-check on the real AR-100k
npz+JSON pair. `check.py` 212 green ×2. **BOX ENDGAME BABYSAT
THROUGH THE BOUNDARIES: A-s0 COMPLETED 40k** (formal final probe
**7.0882@40k**, gate <9 passed with margin; step_040000 saved;
chained panel eval confirmed scoring on GPU0 — 832/25.8k frames at
01:28Z); B's eval @11.7k/25.8k on pacing (read ~03:1xZ); **s1/s2
COMPLETED 40k at the babysat ~01:35Z boundary — formal final
probes 6.9444 / 7.0231 @40k (gate <9, passed with margin), both
`step_040000` saved, and ALL FOUR panel eval chains confirmed live
at 01:36Z** (4 eval procs; s1/s2 in load phase). The full batch —
every arm trained, every gate passed, every eval chained — closes
its training phase with zero interventions. Draws run 3 @21k/25.8k
(~01:50Z, then runs 4–5). Discord: no new messages; 👍 on the
SnapFlow-complete post (recorded). Queue depth: box → E4B (GPU-side
only, needs σ_seed from THIS instrument's output); local → fairness
probe → SnapFlow distill (launch-ready) — ≥2 ✓. GPUs busy + CPU
queue non-empty (results post ~03–04Z runs the instrument, E4B
GPU-side checklist) → `run_work_next` armed per no-idle-pauses.*

*Previous update 2026-08-06 00:57–01:1xZ (real `date -u`) — work session: **RESUME
HARDENING LANDED (#18.4, deep-dive finding 2 — all three traps closed
before the E4B 100k launch opens its crash+resume risk window; idea
#3 longer-training unblocked).** (a) The fresh-seed-on-resume
convention is now ENFORCED, not assumed: `--resume` with the
checkpoint's recorded `train_args.seed` dies loud at startup —
before data/model build, all ranks — because the epoch-0 restart
replays exactly the batches and τ/ε draws already trained on;
`--allow-same-seed-resume` is the explicit reproduction-only escape
hatch (parse-guarded to `--resume`), and checkpoints predating
train_args recording warn instead of dying. (b) Live-backbone
resume now prints an honest WARNING: fp32 masters restart snapped
to the bf16 grid (sub-bf16 updates discarded at every boundary —
masters are never serialized); the stale "lossless continuation"
comment in `save_checkpoint` corrected to frozen-backbone-only.
(c) The resume hyperparameter note covers EVERY optimizer param
group (was group 0 only — a changed `--backbone-*-lr` on resume was
silently ignored): CLI intent captured per group at construction,
compared against restored `initial_lr` so a schedule-decayed lr
can't fake a mismatch. 11 new tests (`tests/test_resume_guards.py`),
`check.py` green (212); live oracle on the real flow-80k
`step_080000`: same-seed refused / fresh-seed proceeds in order.
Coupling handled: `snapflow_recipe_verify` POST_TEACHER_DEFAULTS
extended (new TrainArgs field at inert default), stage 0 re-run
green (51 fields verbatim, 11 deltas). Babysits 01:07/01:10Z: box
×4 healthy — **all three control probes now sub-7** (A-s0
6.955@39k, s1 6.926@37k, s2 6.973@37.5k), A-s0 @39.5k hits 40k
~01:14Z, s1/s2 @37.5–37.6k ~01:26Z, 0.39–0.40 s/step, grad norms
nominal; B's chained panel eval LIVE @3.9k/25.8k frames (~175
frames/min ⇒ read lands ~03:1xZ, controls' evals queue behind their
40k boundaries). Draws run 3 @16.8k/25.8k on pacing (~01:50Z, then
runs 4–5). **OWNER EXCHANGE 01:11Z (replied ~01:15Z):** owner asked
whether the draws runs generated an HTML eval report and wanted to
see the mean-of-10 charts — answered yes (`bijou.eval --report`
writes self-contained HTML per run) and **sent the draws-10 report
itself as a Discord attachment, landing `discord.py post --attach`
(≤10 MB multipart upload) within the exchange** to do it; caveat
stated honestly: the report's charts show the post-average
(mean-of-10) prediction — per-draw spaghetti needs the
`--dump-draws` npz from the fairness probe (~06–09Z), overlay
figures promised for the results post. Stage-2b still awaiting
owner steer. Queue depth: box → E4B (GPU-side only); local →
fairness probe → SnapFlow distill (launch-ready) — ≥2 ✓. GPUs busy
+ CPU queue non-empty (box results post when the four panel reads
land, E4B GPU-side checklist) → `run_work_next` armed per
no-idle-pauses.*

*Previous update 2026-08-06 00:55–00:57Z (real `date -u`) — tick: **B COMPLETED 40k
AND ITS PANEL EVAL CHAIN FIRED — watched through the boundary.** B
(aux-off) finished at ~00:54Z: final loss 3.307, formal final probe
**7.702@40k** (gate <9, passed with margin), `step_040000` saved and
the chained panel eval confirmed live on it (correct k4l2 plan +
`--dump-predictions`, eval log `eval_fontaine_arb_rcond_auxoff_40k_1xh100_40k.log`)
— babysat in-session via a background watcher on save-dir + eval-pgrep
rather than exiting blind at the boundary. Controls healthy and in
their endgame: A-s0 @38.0k (probe 7.21@37.5k), s1 @36.3k (7.06@36k —
batch best), s2 @36.3k (7.08@36k), 0.38–0.41 s/step, grad norms
nominal — they hit 40k ~01:10–01:25Z and auto-chain their own panel
evals; reads land ~01–02:3xZ. Draws run 3 @14.3k/25.8k, 99% util, on
pacing. Discord: no new messages, no new reactions (history-checked);
stage-2b still awaiting owner steer. Boxsync loop alive; local disk
1.4 T free. Queue depth: box → E4B (GPU-side only); local → fairness
probe → SnapFlow distill (launch-ready) — ≥2 ✓. GPUs busy + CPU queue
non-empty (box results post when the four panel reads land
~01–02:3xZ, then E4B GPU-side checklist) → `run_work_next` armed per
no-idle-pauses.*

*Previous update 2026-08-06 00:26–00:5xZ (real `date -u`) — work session: **SNAPFLOW
DISTILL IMPLEMENTATION COMPLETE — all five pre-launch checklist items
landed in one session; the launch path is now zero-CPU and gate (a)
is already PASSED on the real checkpoint**
([pre-reg](posts/2026-08-06-prereg-snapflow-distill.md), ideas #12).
(1) **φ_s target-time embedding** in `FlowDecoder` behind
`target_time_embed` (two-layer MLP mirroring the τ path, output
zero-init ⇒ inert until trained), serialized through
`bijou_config.json` with absent-key back-compat; `--init-from` gains
a sanctioned "φ_s extension" branch (config guard allows exactly the
False→True direction; loader tolerates exactly the four fresh φ_s
keys — reverse direction and any other diff stay hard errors). (2)
**`bijou.train --distill snapflow`**: `L = α·L_FM +
(1−α)·λ·L_shortcut` with α=0.5/λ=0.1 frozen as code constants,
stop-gradient two-step-Euler shortcut targets at the pure-noise end
(x_mid = ε − ½·sg F(ε,1,1); v_target = ½[sg F(ε,1,1) + sg
F(x_mid,½,½)]; grad forward at s=0), one shared prefix encode, both
mean- and sum-form (chunked backward stays available); flow-only
guards; `--distill snapflow` implies the embedding. (3) **1-NFE eval
switch**: `bijou.eval --target-time {t,zero}` — loud, never inferred
from step count, refused on non-φ_s checkpoints, threaded through
single-draw AND `--sample-draws` paths, recorded in report JSON +
npz scalars + banner. (4) **Oracles: 10 new tests** (extension adds
exactly the φ_s keys; zero-init identity bit-exact incl. s=0;
loss ≡ frozen mix with closed-form zero-field value; sums
reconstruct mean; 1-NFE sampling ≡ ε − F(ε,s=0,t=1); config
round-trip; guard direction test) — `check.py` green (201). **Gate
(a) EXECUTED on the real flow-80k step_080000: 6/6 forwards
bit-exact, PASSED** (`fontaine/scripts/snapflow_identity_oracle.py`,
CPU). (5) **Launcher staged + diff-verified through the real
`parse_args`** (`fontaine/scripts/launch_local_snapflow_distill_30k_1xh100.sh`,
copies in `~`): stage-0 recipe verify proves launcher == teacher
train_args + pre-registered deltas ONLY (50 fields verbatim, 11
deltas: steps 30k, LR 2.5e-5, clip 1.0, B24 1×GPU, init-from,
distill flags, bookkeeping); stages chain gate (a) → gate (b)
(step-0 extended checkpoint materializer +
`snapflow_drift_gate.py` vs the banked flow npz, needs GPU) →
training → endpoint 1-NFE panels (draws 1/5/10). @10k record-only
probe staged (`probe_snapflow_10k_1nfe.sh`) with the charter §3
no-co-location note: runs on a quiet GPU (box push or retro), kill
line is catastrophic-only; in-run eval_chunk_mae is the live watch.
Launch waits ONLY on: local GPU quiet (draws chain + fairness probe
~06–09Z) + the σ_draw finalization amendment (draws runs 3–5).
Babysits 00:36/00:52Z: box ×4 healthy — B @39.7k (~00:54Z hits 40k,
auto-chains panel eval; total 3.27, probe 7.679@38k well under
gate), A-s0 @37.5k (action 3.28), s1 @35.7k, s2 @35.8k, 0.38–0.40
s/step, grad norms nominal; controls done ~01:0x–01:3xZ. Draws run 3
@13.8k/25.8k on pacing. No Discord traffic; stage-2b still awaiting
owner steer. Queue depth: box → E4B (GPU-side only); local →
fairness probe → SnapFlow distill (NOW launch-ready) — ≥2 ✓. GPUs
busy + CPU queue non-empty (box results post when panel reads land
~01–02:3xZ, E4B GPU-side items) → `run_work_next` armed per
no-idle-pauses.*

*Previous update 2026-08-06 00:25Z (real `date -u`) — tick: **both chains healthy;
B ~20 min from 40k.** Box ×4: A-s0 @34.9k, B @36.9k, s1 @33.0k, s2
@33.1k — 0.38–0.40 s/step, util 58–100%, ~71–75 GiB, grad norms
nominal. Probes all stepping down well under the closed gates: A-s0
7.368@34.5k, s1 7.417@33k, s2 **7.069@33k** (batch best), B
7.738@36.5k. B total 3.17–3.33@36.9k vs control actions 3.27–3.43 —
line noise, read unchanged. B hits 40k ~00:47Z and auto-chains its
panel eval (no decision pending at the boundary — tick exits rather
than babysitting); controls ~01:0x–01:3xZ. Draws run 3 (draws=5)
@9.0k/25.8k, 96% util, on pacing. Discord: no new messages, no new
reactions (history-checked); stage-2b still awaiting owner steer.
GPUs busy + CPU queue non-empty (box results post when panel reads
land ~01–02:3xZ, SnapFlow impl checklist items 1–5, E4B GPU-side
items) → `run_work_next` armed per no-idle-pauses.*

*Previous update 2026-08-06 00:14–00:3xZ (real `date -u`) — work session: **SNAPFLOW
1-NFE SELF-DISTILLATION PRE-REGISTERED (#12) — the local-GPU queue
slot after the draws chain + fairness probe is now filled**
([pre-reg](posts/2026-08-06-prereg-snapflow-distill.md)). The
session's one item = the sanctioned lit slice taken as a *targeted*
deep-read (SnapFlow full recipe + the two flagged unread pointers),
feeding straight into the pre-reg: SnapFlow (2604.05656) recipe
frozen — α=0.5/λ=0.1 loss mix, stop-gradient two-step-Euler shortcut
targets (no EMA teacher), zero-init φ_s target-time embedding (only
new params; step-0 model ≡ teacher exactly — that's the hard
validation oracle), 30k steps LR 2.5e-5 cosine/500 warmup, trunk
frozen, ~12–20 h 1×H100. Subject flow-80k `step_080000` (verified
local); primary read = full panel at 1-NFE vs 6.6232 with
+max(3σ_draw, 0.15) band (σ_draw via finalization amendment from
draws runs 3–5); **deployment headline read: mean-of-10@1-NFE vs the
AR anchor 5.8026 at ~one-Heun-5-draw cost — if it holds, the charter
§2 "unconstrained class" caveat on the draws win closes** (this is
the owner's 21:48Z pre-stated branch, executed). Feasibility
verified pre-post: euler solver + `sample_actions(noise=…)` +
cosine/warmup + `--init-from` all native; impl checklist (φ_s,
`--distill snapflow`, loud 1-NFE eval switch, oracles) = queued CPU
items. Pointer reads closed into ideas: OFP (2603.12480) banked as
the reserve recipe; GoldenStart screened out (online-RL setting);
**Golden Ticket (2603.15757) banked in #1 — single searched noise
vector, inference-only, gains grow at fewer steps; our panel gives
the offline search criterion the paper lacks** (pairs with 1-NFE +
mean-of-N; needs its own pre-reg). `check.py` green (191). Babysits
00:2xZ: box ×4 healthy — A-s0 @34.5k (action 3.34), B @36.5k (total
3.42 — single-line read *above* controls' action 3.29–3.34 this
poll: the margin keeps oscillating at line noise, read unchanged),
s1 @32.6k, s2 @32.8k, 0.39–0.41 s/step, grad norms nominal; B hits
40k ~00:45Z then auto-chains its panel eval, controls ~01:0x–01:3xZ.
Draws run 3 @8.5k/25.8k, 96% util. No Discord traffic; stage-2b
still awaiting owner steer. Queue depth after this session: box →
E4B (pre-registered, GPU-side items only); local GPU → fairness
probe (pre-registered) then SnapFlow distill (pre-registered,
impl checklist pending) — ≥2 ✓. GPUs busy + CPU queue non-empty
(box results post when panel reads land ~01–02:3xZ, SnapFlow impl
checklist items 1–5, E4B GPU-side items) → `run_work_next` armed
per no-idle-pauses.*

*Previous update 2026-08-06 00:13Z (real `date -u`) — tick: **both chains
healthy; B ~28 min from 40k completion.** Box ×4: A-s0 @33.5k, B
@35.5k, s1 @31.8k, s2 @32.0k — 0.38–0.41 s/step, grad norms nominal,
~71–75 GiB (GPU0's 0%-util sample is the known between-batch/eval
idle; log advancing). Probes all stepping down well under the closed
gates: A-s0 **7.212@33k**, s1 7.333@31.5k, s2 7.203@31.5k, B
7.844@35k. B total 3.27–3.34@35.5k vs control actions 3.32–3.44 —
line noise, unchanged read. B hits 40k ~00:42Z and chains its panel
eval automatically (no decision pending at the boundary — tick exits
rather than babysitting); controls ~01:0x–01:3xZ. Draws run 3
(draws=5) @6.4k/25.8k, 99% util, on pacing. `boxsync_loop` v2 alive
and marking. Discord: no new messages, no new reactions
(history-checked); stage-2b still awaiting owner steer. GPUs busy +
CPU queue non-empty (box results post when panel reads land
~01–02:3xZ, E4B GPU-side checklist items, stage-2b pending steer) →
`run_work_next` stays armed per no-idle-pauses.*

*Previous update 2026-08-06 00:03–00:1xZ (real `date -u`) — work session: **E4B
PRE-LAUNCH CHECKLIST ITEM 6 DONE — the rsync-back loop is extended and
hardened; every CPU-side item on the E4B launch path is now closed**
(pre-reg [checklist](posts/2026-08-05-prereg-e4b-screen.md); launch
waits only on GPU-dependent items: box free, memory smoke,
finalization amendment σ_seed, box push+checkout ≥cb51f74 after the
four chained panel evals). `~/boxsync_loop.sh` v2, deployed +
restarted in `fontaine-rsync`: (1) `fontaine_arb_rcond_e4b_100k_ddp4`
added to RUNS (its train/eval logs already matched the log globs);
(2) **partial-copy guard** — a step dir gets `.synced_complete` only
when a follow-up `--dry-run` transfers nothing, so the local E4/E5
panel evals can refuse a mid-save copy; (3) **panel-step repair** —
E4B steps {25k, 50k, 100k} re-sync until marked complete even after
leaving the latest-2 window; (4) **local rotation, E4B only** —
keep latest two + panel steps, prune the rest. The rotation is
load-bearing disk math: E4B saves ≈ 35–40 GB × 40 ≈ 1.5 T unpruned,
and local free is exactly 1.5 T — the unmodified loop (which keeps
everything it ever synced) would have filled the disk mid-run; the
four 40k-run local copies are never pruned. Verified before deploy:
`bash -n` + a sandboxed one-pass run (HOME redirected, ssh/rsync
mocked via PATH shims, pre-seeded stale step dirs) proved marker +
prune + keep behavior; also confirmed the existing loop's
`sort | tail -2` is CORRECT (zero-padded step dirs) — checked before
"fixing" it. First real pass is live and marking actual checkpoints.
Strand-proofing: `boxsync_loop.sh`, the E4B DDP4 launcher, the B12
smoke script, and all four 40k launchers copied off the temporary box
into `fontaine/scripts/box/` (they defined pre-registered runs and
existed only on hardware that can vanish). `check.py` green (191).
Babysit 00:1xZ: box ×4 healthy — A-s0 @33.2k, B @35.3k, s1 @31.5k,
s2 @31.6k, 0.38–0.40 s/step, grad norms nominal; B total
3.26–3.30@35.3k vs control actions 3.29–3.45 — line noise; B done
~00:40Z then chains its panel eval, controls ~01:0x–01:3xZ. Draws
run 3 (draws=5) @5.8k/25.8k, util healthy. No Discord traffic;
stage-2b still awaiting owner steer. GPUs busy + CPU queue non-empty
(box results post at arm completion ~00:40–02Z, E4B GPU-side items,
stage-2b pending steer) → `run_work_next` armed per no-idle-pauses.*

*Previous update 2026-08-06 00:02Z (real `date -u`) — tick: **both chains healthy;
box endgame on schedule.** Box ×4: A-s0 @32.4k, B @34.5k, s1 @30.7k,
s2 @30.9k — 0.38–0.40 s/step, util 57–85%, grad norms nominal. Probes:
**controls' formal 30k gate reads all in and PASSED** — s1 7.842@30k,
s2 7.206@30k (A-s0's passed last session; B's formal 8.178 earlier) —
every pre-registered probe gate in the batch is now closed. Latest:
A-s0 7.557@32k, B 7.916@34k, s1 7.350@30.5k, s2 7.221@30.5k. B total
3.30@34.5k vs control actions 3.33–3.47 — margin line noise, as read.
B hits 40k ~00:40Z then chains its panel eval; controls ~01:0x–01:3xZ.
Draws run 3 (draws=5) @4.2k/25.8k, 100% util, on pacing. Discord: no
owner traffic (only our stage-2 results post), no new reactions
(history-checked); stage-2b still awaiting owner steer. GPUs busy +
CPU queue non-empty (box results post at arm completion ~00:40–02Z,
E4B GPU-side checklist, stage-2b pending steer) → `run_work_next`
armed per no-idle-pauses.*

*Previous update 2026-08-05 23:37–00:0xZ (real `date -u`) — work session: **STAGE-2
SIGN PROBE EXECUTED — THE PRE-REGISTERED ESCALATION BRANCH FIRED: 3 of
4 reference populations are NOT sign-consistent, so no candidate cell
opened and no mirror verdict ships**
([results post](posts/2026-08-05-sign-stage2-results.md), probe
`probes/probe_sign_convention_stage2.py`, report
`~/sign_stage2_results.json`, 38-repo flow cache
`~/sign_stage2_cache/`). Instrument built + run exactly per the frozen
pre-reg (CPU-only, nice-19 ×8 beside both live GPU chains): population
gate — wrist_roll 9/15 agree (median ρ +0.16), wrist_flex 10/15
(−0.13), shoulder_lift 9/15 (−0.46), **only shoulder_pan VALID 13/15
(−0.24)**; synthetic-flip oracle on the valid t_x family **PASSED
end-to-end** (original NORMAL / doctored MIRRORED, bootstrap mass
1.000 both ways, ρ ∓0.887) — mechanism works where the population
premise holds, so the failure is the premise, not the machinery.
Diagnosis shipped with the claim: image-plane statistic signs follow
*camera mounting* (the two cams sign-disagree in 11/15 shoulder_lift
refs at |ρ| up to 0.85; ego-cam rule NO-MARGIN on ~half; ω is
wrist-cam-only — 2/15 refs reach |ρ|≥0.3 off-wrist) — NOT evidence
that joint conventions vary corpus-wide. #13 updated: three stage-1
mirror cells remain unresolved leads, repair arm neither eligible nor
dead; **proposed stage-2b posted to Discord for owner steer:
re-pool references per (dim, camera kind) via the corpus's
`meta/camera_kinds.json`** (2026-08-02 VLM labeling pass) + label-gated
ego rule, reusing today's flow cache (~cheap). Figure: population
strip plot (dataviz-skill compliant, reference-palette slots).
`check.py` green (191); blog built + Space pushed; Discord posted
(no owner traffic). Babysits 23:52/00:00Z: box ×4 healthy — A-s0
@32.2k, B @34.2k (done ~00:30Z), s1 @30.5k, s2 @30.6k, 0.38–0.40
s/step, **all four 30k probe gates passed** (A-s0 7.45@31k, s1
7.71@29.5k, s2 7.52@29.5k, B 8.178 formal); draws run 3 (draws=5)
scoring 3.7k/25.8k @94% util. GPUs busy + CPU queue non-empty (box
results post ~00:30–02Z, E4B GPU-side checklist, stage-2b pending
owner steer) → `run_work_next` armed per no-idle-pauses.*

*Previous update 2026-08-05 23:36–23:4xZ (real `date -u`) — tick: **all four box
arms healthy in the endgame stretch; draws run 3 (draws=5) confirmed
past load.** Box ×4: A-s0 @29.7k, B @31.6k, s1 @28.1k, s2 @28.3k,
all 0.387–0.393 s/step, util 60–96%, grad norms nominal. Probes:
A-s0 7.91@29.5k, B 8.118@31.5k, s1 7.845@28k, s2 7.796@28k — all
well under the gate values (B's formal 30k gate passed 8.178 last
tick; controls hit their own 30k reads in ~2k steps, tracking
~1-point under). B total 3.31–3.37@31.6k vs control actions
3.38–3.50 — margin still line noise. B done ~00:30Z, controls
~01:0x–01:3xZ. Draws run 3: checkpoint loaded, first-poll util rule
applied (post-load util confirmed before exit — see below). No
Discord traffic, no new reactions (history-checked). GPUs busy +
CPU queue non-empty (box results post ~00–02Z, E4B GPU-side
checklist, stage-2 probe implementation) → `run_work_next` armed
per no-idle-pauses.*

*Previous update 2026-08-05 23:12–23:4xZ (real `date -u`) — work session: **STAGE-2
SIGN-CONVENTION PRE-REG POSTED (the queue's next CPU item) + DRAWS
RUN 2 LANDED A HEADLINE: mean-of-10 flow BEATS the AR-100k panel
anchor.** (1) [Pre-reg](posts/2026-08-05-prereg-sign-stage2.md)
freezes stage 2 of ideas #13 before any probe code exists: optical-flow
cross-check on the three stage-1 mirror-signature cells
(dishTidyUp_anomaly wrist_flex, groceriesSorting_expert wrist_roll,
aractingi shoulder_lift), CPU-only ~20–40 min. Frozen: Farneback
params, isolated-motion pair selection (0.5°/frame, 2× dominance),
ego-cam identification (cams are unlabeled `image`/`image2`),
15-repo so100 reference population + 80% sign-consistency validity
gate, bootstrap MIRRORED/NORMAL/INCONCLUSIVE rules, synthetic-flip
hard validation gate, Dongkkka + kevin510 as pre-declared specificity
controls, and a stream-consistency read separating calibration-mirror
from action-only flip. Feasibility verified pre-post: repos local,
torchcodec decodes the AV1 videos, parquet streams intact. Execution
= a later session; ≥1 MIRRORED ⇒ repair arm gets its own pre-reg.
Also: stale `posts/index.md` refreshed (was 12 posts behind SUMMARY).
`check.py` green (191); blog built + Space pushed. (2) **Draws run 2
(draws=10 heun-30) finished 23:31Z: chunk_mae 5.365 / first_mae
1.424** vs single-draw 6.6232/1.9331 (−19%/−26%) and vs the AR-100k
anchor **5.8026**/2.1431 — mean-of-10 flow now beats AR on BOTH panel
columns (unconstrained class: 10× NFE; charter §2 caveat stands until
distilled — SnapFlow is the queued leg). Banked prediction check
(ideas #1): "chunk_mae moves a lot" ✅, "first_mae barely" ❌ —
first_mae moved 26%; honest miss recorded in #1, full analysis in the
results post after runs 3–5 + the fairness probe. Run 3 (draws=5)
chained 23:32Z, in load phase — next tick confirms post-load util.
Babysit 23:31Z: box ×4 @27.7–31.2k, 0.40–0.41 s/step, all probes
under gate (B formal 30k gate passed 8.178 last tick); B total
3.408@31.2k. B done ~00:3xZ, controls ~01:0x–01:3xZ. No Discord
traffic. GPUs busy + CPU queue non-empty (box results post ~00–02Z,
E4B GPU-side checklist, stage-2 probe implementation) →
`run_work_next` armed per no-idle-pauses.*

*Previous update 2026-08-05 23:22Z (real `date -u`) — tick: **B's FORMAL 30k
PROBE GATE READ IS IN: 8.178@30,000 — PASSED** (kill line was >9;
cross-check vs the 25k panel moot at this margin). All four arms
healthy: box ×4 @27.0–30.1k, **s/step recovered to 0.38–0.44**
(last tick's 0.50–0.52 watch item resolved — it was the parity-job
CPU contention, now ended), util 55–95%, grad norms nominal. Probes:
B 8.178@30k (its 29.5k read 8.338), A-s0 8.203@28k, s1 7.731@26.5k
(batch best), s2 7.899@26.5k — controls hit their own 30k gate reads
in ~3.5k steps, all tracking well under. B total 3.31–3.41@30.1k vs
control actions 3.43–3.55 this poll — margin still line noise at the
endpoint. B hits 40k ~00:3xZ, controls ~01:0x–01:3xZ. Draws run 2
@24.8k/25.8k — done within minutes, chains to run 3. No Discord
traffic; no new reactions (🔥 on the reaction-rendering post already
recorded). GPUs busy + CPU queue non-empty (box results post
~00–02Z, E4B GPU-side checklist, stage-2 sign pre-reg) →
`run_work_next` armed per no-idle-pauses.*

*Previous update 2026-08-05 23:12–23:2xZ (real `date -u`) — work session: **E4B
PRE-LAUNCH CHECKLIST ITEMS 2+3 DONE + LAUNCHER/SMOKE STAGED — the
launch path now waits only on GPU-dependent items** (pre-reg
[checklist](posts/2026-08-05-prereg-e4b-screen.md)). Executed during
the GPU-busy window, all CPU/network-only, ionice'd: (1) **checkpoint
staged** — `google/gemma-4-e4b-it` (15 G, snapshot `ee0ef60`)
downloaded into the box HF cache in ~1 min (item 2 ✅). (2) **Parity
spot-check PASSED on the box, CPU** — full `bijou.gemma4.verify_parity`
harness: greedy token ids **bitwise OK on every text and image case**;
logit-level "within tol" spreads (max|Δ| ≤1.66) with token agreement =
the harness's documented E4B ULP-tie behavior (item 3 ✅, log
`~/e4b_parity.log`). (3) **DDP4 launcher staged + diff-verified**
(`~/launch_box_fontaine_arb_rcond_e4b_100k_ddp4.sh`): diff vs the
mainline `launch_arb_rcond_100k.sh` shows ONLY the pre-registered
deltas — `--backbone` E4B, B10→B12 (the recipe's launch value; 10 was
the post-OOM resume edit), `${CHUNK_ARGS}` hook, run naming — no
science flag differs; `BACKWARD_CHUNKS` is a required env var so the
launcher refuses to run before the finalization amendment picks the
rung; E1/E2/E3 gates in the header; chains the E5 endpoint 4-GPU
sharded panel with dumps. (4) **Memory-smoke script staged**
(`~/smoke_e4b_b12.sh [chunks]`: 60 steps 1×GPU B12, 2-s VRAM sampler
prints peak, rung semantics in header). **Remaining before launch,
all blocked on tonight's arms:** box free (~02:30–03:15Z with evals);
smoke (needs 1 free GPU); finalization amendment (σ_seed from the
replicate panels); rsync-back extension; **and push+checkout the box
to ≥cb51f74 (chunked backward — box is at cc0b922) strictly AFTER all
four chained panel evals finish** (no code swap under a pre-registered
eval). Babysit 23:2xZ: box ×4 @26.3–29.7k, probes A-s0 8.334@27.5k /
B 8.338@29.5k / s1 8.11@26k / s2 7.898@26k — all under the 30k gate
value, B's formal 30k read next eval; B total 3.364@29.7k. **Watch
item: s/step 0.50–0.52 ×4** (was 0.39–0.44; still inside the 0.4–0.7
band — likely CPU contention from the parity job, now finished; next
tick verifies recovery). Draws run 2 @24.4k/25.8k, ~done, chains to
run 3. No Discord traffic. GPUs busy + CPU queue non-empty (box
results post ~00–02Z, E4B GPU-side checklist, stage-2 sign pre-reg)
→ `run_work_next` armed per no-idle-pauses.*

*Previous update 2026-08-05 23:11Z (real `date -u`) — tick: **both chains
healthy; B one probe from the formal <9@30k gate read.** Box ×4
@25.8–29.1k, 0.39–0.41 s/step, util 71–99%, grad norms nominal:
probes A-s0 8.17@27k, B 8.34@29k, s1 **7.77@25.5k** (batch best),
s2 8.69@25.5k (noisy bounce off 7.89@25k, within the ±0.5–0.8
band) — all four still under the gate value; **B aux-off total
3.40@29.1k vs control action losses 3.41–3.47 — the margin has
converged to line noise at the endpoint approach** (consistent with
the mainline E4 "within noise" read after the transient early
lead). B hits 40k ~00:4xZ, controls ~01:0x–01:3xZ. Draws run 2
@23.7k/25.8k — done ~23:30Z, chain rolls to runs 3–5. Discord: no
new messages; no new reactions beyond the recorded 🔥. GPUs busy +
CPU queue non-empty (box results post ~00–02Z, E4B launch
checklist, stage-2 sign pre-reg) → `run_work_next` armed per
no-idle-pauses.*

*Previous update 2026-08-05 22:43–23:1xZ (real `date -u`) — work session:
**CHUNKED BACKWARD LANDED (`--backward-chunks` in `bijou.train`) — the
E4B B12-OOM fallback is now ready BEFORE the memory smoke, and it
surfaced a mechanism error in the E4B pre-reg, corrected by
[Amendment 1](posts/2026-08-05-prereg-e4b-screen.md) before any E4B
data exists.** The pre-reg's "equal chunks ⇒ mean of chunk-means =
batch mean" is FALSE for token-weighted CE pooling (unequal FAST
token counts per sample); the implementation is stronger: per-chunk
SUM-form losses over FULL-step normalizer counts (data-only
pre-pass; aux ratio over the global aux count; DDP no_sync until the
last chunk; `static_graph` dropped when chunking) — exactly the
unchunked gradient even with unequal counts, up to fp reduction
order. Oracles all run pre-post: chunking OFF **all three CPU loss
oracles bit-exact** (2.7903/1.9152, 4.9232/4.8631,
27.8262/27.7701); ar_fast chunked CLI A/B **bitwise** at printed
precision; ar_backbone chunked A/B loss-identical with a 0.28%
grad_norm delta that was **diagnosed, not waved off** (three-way
experiment: bit-identical sliced memory ⇒ gradients rel ~5e-7 — the
math is exact; the residual is per-chunk collation width shifting
prefix-encode fp reduction order, amplified by the random tiny
fixture's saturated 262k softmax). 7 new tests incl. the
unequal-aux-counts gradient-equivalence oracle (rel < 1e-5);
`check.py` green (191); `docs/architecture.md` §5 documents the
mechanism. Babysit 23:0xZ: box ×4 @25.2–28.5k, 0.39–0.40 s/step,
util 57–100%: **probes stepped down a leg — s1 7.84@25k, s2
7.89@25k, A-s0 8.14@26k, B 8.33@28.5k — all four now BELOW the
<9@30k gate value outright**; B total 3.40@28.5k still below every
control's action loss; B hits 40k ~00:4xZ, controls ~01:0x–01:3xZ.
Draws run 2 @23.1k/25.8k (ETA ~23:5xZ, then runs 3–5). Discord: no
new messages. GPUs busy + CPU queue non-empty (box results post at
~00–02Z + E4B launch checklist, stage-2 sign pre-reg) →
`run_work_next` armed per no-idle-pauses.*

*Previous update 2026-08-05 22:41–22:45Z (real `date -u`) — tick: **both
chains healthy; all four box arms now clearly under the <9@30k gate
track.** Box ×4 @23.0–26.0k, 0.38–0.40 s/step, util 48–93%, ~71–74
GiB, grad norms nominal: A-s0 total 3.74@24.5k (action 3.47), s1
3.85@23k (action 3.57), s2 3.84@23.3k (action 3.58), **B aux-off
total 3.436@26k — still below every control's action loss.** Probes:
A-s0 8.35@24k, s1 8.33@23k, B 8.41@25.5k, s2 9.16@23k (noisy bounce
off its 8.28@22.5k, within the ±0.5–0.8 band). Draws run 2
@20.5k/25.8k on pacing (ETA ~23:45Z, then runs 3–5); the 0%-util
sample is the known between-batch idle, scored-frames advancing.
Discord: no new messages; history check surfaced a **🔥 on our
reaction-rendering post** (positive ack, recorded — the new
history-check protocol caught its first reaction). The 22:31–22:39Z
exchange stands settled (ladder approved, polling decision
owner-acked); channel watched a further ~5 min of silence before
exit. ~4–6k steps to the 30k probe gates; first arm completions
~00–02Z. GPUs busy + CPU queue non-empty (chunked-backward impl if
the B12 smoke needs it, stage-2 sign pre-reg, box results post
~00–02Z then the E4B launch checklist) → `run_work_next` armed per
no-idle-pauses.*

*Previous update 2026-08-05 22:20–22:40Z (real `date -u`; NB this session's
commit labels "23:0x/23:1x" ran ~25 min ahead of the real clock —
label skepticism stays warranted) — work session: **E4B SCREEN
PRE-REG POSTED (the owner-picked item) + the overdue lit slice
taken, which produced a fourth pre-declared fairness read — then a
LIVE OWNER EXCHANGE (22:31–22:35Z) resolved the fallback ladder and
landed two more items.** Exchange: (a) owner asked whether
B12-doesn't-fit means no E4B run — clarified the ladder (chunked
backward at loader-B12 IS the expected path; no-launch is only the
low-single-digit-% bottom rung where even 3-sample chunks OOM, and
any workaround there breaks matched-params ⇒ owner call); **owner
22:35Z: "Noted on the batch size ladder, I agree with your
strategy" + 👍 — the ladder is owner-approved.** (b) Owner asked
about Discord capabilities: answered (attachments yes via CDN URL;
reactions were invisible) and **landed reaction rendering in
discord.py within the exchange** (read+history now print
`reactions:`; verified live on the owner's own 👍; polling caveat
documented — reactions surface within a tick, not instantly).
Follow-up 22:38–22:39Z: owner asked if we can beat polling —
decision (delegated "up to you"): **keep REST polling, no gateway
daemon** (detection ≠ response; sessions are the response floor
either way; charter run-only-what-changes-the-next-decision), with
ticks now `history`-checking recent posts so late reactions surface
within a tick (tick.md updated). **Owner 22:39Z: "I'm ok with 10m
delay fwiw" — settled.** (c)
**`read4_energy_score` LANDED + VALIDATED** per Amendment 2 before
any probe data exists (degenerate draws=1: interaction exactly 0,
ES ≡ direct RMS-L2, 6.6232 anchor intact, 6/6 checks OK; AR +
banked-flow baselines join on the probe rows) — all four fairness
reads are now execution-ready for the probe at ~06–09Z.*

*Same session, earlier (~22:20–22:35Z):
(1) [E4B pre-reg](posts/2026-08-05-prereg-e4b-screen.md): verbatim
mainline recipe + `--backbone google/gemma-4-e4b-it` (AR path
verified fully config-driven — full-depth trunk, tail-anchored block
base adapts, no expert/stream surface); eff-48 as owner-picked with
a pre-registered **chunked-backward fallback ladder** (B12 direct →
2×6 → 3×4 → 4×3 at loader-B12, gradient ≡ B12 up to fp reduction
order) because `bijou.train` has **no grad-accum today** and E2B B12
peaked 77.5 GiB with E4B text ~2.2× params — the impl + oracles is a
pre-launch CPU item if the memory smoke OOMs. Both seams stated up
front: the E2B reference's own 48→40@20k batch seam (E4B holds +15%
samples ⇒ kills conservative-valid, adopts carry the caveat) and the
Δ19-episode probe-corpus seam (probe read at ±0.5 floor; the k4l2
panel — scored on this box copy by the owner today — is the seam-free
decision instrument). Gates: @10k record-only, @30k kill if probe >
7.07 with 25k-panel cross-check, @50k re-check (>6.29), endpoint
adopt iff panel beats 5.8026 by max(3·σ_seed, 0.15) with σ_seed
from tonight's E5 replicate reads via **finalization amendment**.
Launch blocked on: box free + e4b ckpt download (NOT in box cache,
~16 GB) + parity spot-check + B12 memory smoke + amendment. (2) Lit
slice (~20 min, was 2× overdue): **[2606.31382] VLM-to-VLA parameter
redundancy — bigger backbones do NOT consistently help action
performance after adaptation** (banked in #17: the kill branch is a
live outcome; raises #11's prior), and **Energy Policy [2510.12483]
→ [Amendment 2](posts/2026-08-05-draws-fairness-amendment2.md)
posted before any per-draw data: read 4 = the energy score**
(RMS-normalized, valid-element mask, proper scoring rule — the
principled middle between MAE and the best-of-N oracle; candidate
distributional column for the comm holdout). `read4_energy_score` +
degenerate validation must land before the probe npz is opened —
queued as the next CPU work item alongside the E4B launch checklist.
check.py green ×3 this session (184). Babysits 22:19/22:30/22:38Z:
box ×4 healthy @21.0–25.7k, 0.38–0.40 s/step (B total 3.44@25.7k —
still below every control action loss; s2 3.727@23k its best line
yet); draws run 2 @20.2k/25.8k on pacing (ETA ~00:1xZ, then runs
3–5). GPUs busy + CPU queue non-empty (~~read4 impl~~ DONE this
session; chunked-backward impl if the smoke needs it, stage-2 sign
pre-reg, box results post ~00–02Z then the E4B launch checklist) →
`run_work_next` armed per no-idle-pauses.*

*Previous update 2026-08-05 22:18Z (real `date -u`) — tick: **both chains
healthy; PROBE <9 NOW TOUCHED BY ALL FOUR ARMS pre-24k — the <9@30k
gate is effectively met batch-wide** (A-s0 8.764@22k its first
sub-9, B aux-off 8.757@23k with a noisy 8.821@23.5k bounce, s1
8.842@20k, s2 8.981@20.5k — B is inside the control probe envelope,
sealing the B-early-lead-was-transient read). Box ×4 @21.0–23.8k,
0.38–0.40 s/step, util 46–99% sampling, ~71–74 GiB, grad norms
nominal: A-s0 total 3.84@22.1k (action 3.57), s1 3.95@21k (action
3.64), s2 3.90@21k (action 3.60), **B aux-off total 3.406@23.76k —
still below every control's action loss.** Draws run 2 @17.95k/25.8k,
log advancing on pacing (ETA ~23:50Z, then runs 3–5; the 0% util
sample is the known between-batch idle). No Discord traffic. ~1.5 h
to the first box arm completions (~5–6.5 h train + eval from 17:12Z
→ reads land roughly 00–02Z). GPUs busy + CPU queue non-empty
(**E4B screen pre-reg is the owner-picked next item**, stage-2 sign
pre-reg, lit slice TWO SESSIONS OVERDUE — must be taken) →
`run_work_next` armed per no-idle-pauses.*

*Previous update 2026-08-05 ~21:51–22:2xZ (real `date -u`) — work session: **THE
MODE-AVERAGING FAIRNESS INSTRUMENT IS EXECUTION-READY — the owner's
21:49Z three pre-declared reads now have a data path**
([Amendment 1](posts/2026-08-05-draws-fairness-amendment.md) on the
noise-draw pre-reg). **Instrument finding en route: the pre-declared
"draws-10 per-draw dumps" could not have existed** — the draws chain
passes no dump flag, and `--dump-predictions` stores the
*post-average* prediction; per-draw chunks died inside
`predict_with_text`. Landed: (1) **`bijou.eval --dump-draws`** —
pre-average `[frames, draws, chunk, dim]` npz + full identity columns
(#18.1 conventions) + standalone scoring-semantics scalars, loud
constraints (needs `--checkpoint` + `--sample-draws > 1`), threaded
through the DDP shard merge; `collapse_draws` factored pure +
unit-tested (dump averages back byte-identical to the prediction —
the mean is still taken once on the full stack). (2) **Probe frozen**:
stride-7 core subset plan (2,458 frames / 792 repos, deterministic
builder) + launcher `~/eval_flow80k_drawsprobe_dump.sh` (draws=10
heun-30, ~30 min 1×GPU, GPU-quiet guard, auto-runs the analysis;
E1-style gate: draw-0 frame-MAE drift vs the banked flow npz < 0.05).
(3) **`fontaine/scripts/draws_fairness.py`** — the three reads with
the report's exact valid-element pooling; joins probe rows to the
banked AR/flow npzs on concat `index` with hard row-agreement asserts.
**Oracles: banked AR-100k panel rebuilt through the edited scoring
path 12/12 cells d=0 (incl. 5.802585); degenerate draws=1 validation
reproduces 6.6232 EXACTLY on reads 1+2 with all-zero dispersion.**
check.py green (184 tests, +5). Launch: first quiet local-GPU
boundary after the draws chain (~06–09Z), before the results post.
Babysits 21:52/22:1xZ: box ×4 healthy @20.3–23k, 0.39–0.42 s/step
(one benign 5.6 s save blip on B; B aux-off total 3.487@23k, still
at/below control action losses 3.63–3.68); draws run 2 @17.2k/25.8k
99% util (ETA ~23:50Z, then runs 3–5). **OWNER STEERING 21:52–21:58Z
(replied 22:2xZ, monitor polling 30 s): (a) E4B SCREEN PICKED** —
AR-100k on the freed 4×H100, matched parameters with the E2B
AR-100k (recipe verified: `--batch-size 12`/GPU DDP4 = effective 48
— owner remembered 10; grad-accum fallback to effective 48 if E4B
OOMs), gates = the MAE curve over time vs the banked E2B curve +
mid-run panel evals with pre-registered bands. **The E4B pre-reg is
the next CPU work item.** (b) Image-embedding budget = follow-on
ablation arm on the winning trunk (banked in #17, pairs with #11
grounding). (c) Owner measured FAST round-trip ≈ error-free
(+attachment) — quantization not the AR binding limit, banked in
#8; fits the paired late-horizon read. GPUs busy + CPU queue
non-empty (**E4B screen pre-reg next**, stage-2 sign pre-reg, lit
slice two sessions overdue) → `run_work_next` armed per
no-idle-pauses.*

*Previous update 2026-08-05 21:47–21:5xZ (real `date -u`) — tick: **both chains
healthy; PROBE GATE <9@30k EFFECTIVELY MET EARLY ON TWO CONTROLS —
s1 8.991@18k, s2 8.982@18k, the first sub-9 probes of the batch**
(A-s0 9.15@18.5k → 9.31@19k noisy bounce; B aux-off 9.58@20.5k — B
is now the *trailing* probe arm despite being ~2k steps ahead,
further strengthening the B-early-lead-was-transient read). Box ×4
@18.2–20.7k, 0.377–0.399 s/step, util 58–62%, ~71–74 GiB, grad norms
nominal: A-s0 total 3.956@19.5k (action 3.642), s1 4.16@18.5k
(action 3.816, one noisy line off 3.906/3.652), s2 4.01@18.2k
(action 3.728), B aux-off total 3.665@20.66k — the action-loss
margin keeps oscillating around zero at line noise (B 3.665 vs
A-s0's action 3.642). Draws run 2 @14.75k/25.8k, 99% util. **LIVE
EXCHANGE: owner 21:48:14Z** (landed seconds after the cursor read)
— challenge on the flow-vs-AR crossover: k≤3 @ 30 fps ≈ 100 ms, not
a realistic replan horizon (inference would need <100 ms). Replied
21:5xZ agreeing with the arithmetic and the thrust: deployable
regime is k≥5 where AR wins today; draws-10 is attribution, not a
deployable config (N draws multiply decode cost); flow's residual
case = first_mae grounding edge + (if draws close the gap) SnapFlow
1-NFE distill + small N; otherwise attribution screens run on the
AR recipe. **Steering applied: weight the AR-side arm in the
limit-attribution plan.** Owner 21:49Z follow-up: **is MAE unfair
to flow — mode-averaging-forgiving?** Replied: yes it's the right
worry and it's measurable tonight on CPU from the draws-10 per-draw
dumps — three pre-declared reads for the results post: (1)
mean-of-draws MAE (ensembling ≈ manufacturing the mode-averaged
predictor; closes gap ⇒ deficit was punished dispersion), (2)
best-of-N MAE (oracle mode-match bound on
'sampled-a-different-valid-mode'), (3) dispersion-conditioned
deficit (the queued unimodality probe — deficit concentrating on
high-disagreement frames = the unfair-penalty signature).
Circumstantial fingerprint already present: flow wins horizon 0–1,
deficit grows with horizon + motion quartile. Honest limit stated:
MAE can't settle actual performance, and the owner's comm-MAE→rig
bridge was built on AR checkpoints — if flow is being punished for
multimodality, the comm holdout needs a distributional column
(best-of-N / energy distance) before it can rank flow arms. Monitor
polling the channel at 30 s while the exchange is live. GPUs busy + CPU queue non-empty (stage-2
pre-reg, lit slice due, E4B screen launcher after box reads) →
`run_work_next` armed per no-idle-pauses.*

*Previous update 2026-08-05 ~21:5xZ (real `date -u`) — work session: **IDEAS #16
INSTRUMENTS LANDED — the rig benchmark is execution-ready up to its
two slots** ([Amendment 1](posts/2026-08-05-prereg-rig-fewshot-benchmark.md)).
Plan frozen (`plans/rig_fewshot_v0_k4l2.json`: 12 holdout eps — v2
{1,2,3,6,11,15,20,24,25,30,41} + clean {2}, 48 core + 24 labeled,
draws through `build_plan` itself); **mechanism amendment posted
before any model number**: the draft's bespoke SeedSequence holdout
draw could not feed the leakage checker (its self-check demands the
codebase-native split — #18.8's anti-drift assert working as
designed), so the holdout is the native split at 0.212/seed 16 =
exactly the pre-registered 11+1 counts. Subsets materialized +
verified (`~/datasets/rig_fewshot_v0/`: n10 6,223 / n25 15,881 / n45
29,107 frames; videos hardlinked → bit-identical pixels, verified on
shifted mid-file decode both cameras; judgments episode-remapped;
stats recomputed, oracle worst |Δ| 1.2e-4 vs both shipped
stats.json). **Leakage certs ×3 PASSED** (first production consumers
of the #18.8 provenance path; doctored-provenance negative control
FAILS loud). **Wrap census CLEAN on both rig repos** (hygiene gate 1
done). Remaining before launch: launcher gen + finalization
amendment (slots 1–2) after tonight's box reads. Babysit ~21:50Z:
box ×4 healthy @17.4–19.8k, 0.38–0.40 s/step (one benign 10.0 s
save-boundary blip on s1), **B aux-off total 3.58–3.60 @19.8k — back
below every control's action loss (3.64–3.79)** after the 21:24Z
margin-zero read; draws run 2 @13.6k/25.8k @100% util. No Discord
traffic. GPUs busy + CPU queue non-empty (launcher gen, stage-2 sign
pre-reg, lit slice due next session) → `run_work_next` armed per
no-idle-pauses.*

*Previous update 2026-08-05 21:24Z (real `date -u`) — tick: **both chains
healthy, no Discord traffic** (only our own #16 pre-reg post). Box ×4
@16.2–18.5k, 0.374–0.382 s/step, util 57–99%, ~71–74 GiB, grad norms
nominal: controls A-s0 total 4.04 @17.3k / s1 4.09 @16.5k / s2 4.08
@16.2k (action 3.64–3.80), B aux-off total 3.686 @18.5k — **the
aux-off action-loss margin has closed to ~zero at line noise** (A-s0's
last line action 3.642 sits below B's 3.686 total; per-line noise
~0.1). Probes now one interleaved band 9.2–10.2: **s1 9.216@16k — new
best across all arms**, A-s0 10.21@17k (noisy bounce off its
9.4472@16.5k), s2 9.93@16k, B 9.80@18k (off its 9.59@17.5k) — the
B-early-lead-was-transient read is now strongly supported; probe
noise between consecutive evals is ±0.5–0.8, so the <9@30k gate is
the next real checkpoint. Draws run 2 @12.4k/25.8k @100% util on the
~5 h pacing. GPUs busy + CPU queue non-empty (#16 follow-on
instruments: subset materializer + plan builder; stage-2 sign
pre-reg; lit slice due) → `run_work_next` armed per no-idle-pauses.*

*Previous update 2026-08-05 ~21:3xZ (real `date -u`) — work session: **IDEAS #16
PRE-REG DRAFT POSTED — the north-star benchmark design is frozen**
([post](posts/2026-08-05-prereg-rig-fewshot-benchmark.md)). Few-shot
rig-transfer v0: sample-efficiency curve MAE(N), N ∈ {0,10,25,45},
over the 57 owner rig episodes — 12-ep fixed holdout
(SeedSequence(16)), nested train subsets as **materialized derived
corpora with the #18.8 leakage gate** (the first consumer of that
work); owner `run_ft_rig.sh` protocol constants, 1×H100 B10,
best-checkpoint-at-200 selection; co-primary chunk_mae + first-4
pooled MAE (k fixed per the flow-vs-AR crossover); 3·σ_ft decision
rule with σ_ft from N25 seed replicates + an honest degrade rule if
σ_ft > 0.5. **Key design find: flow-80k is contaminated as a
few-shot subject** (rig data in its pretrain mix per the owner's
`run_ft_rig_flow.sh` header) — eligibility gate pre-registered;
rcond-100k and all four box arms qualify. Two slots (init selection,
E5 noise scale) fill by finalization amendment after tonight's box
reads; execution ≈ one evening on 1 GPU at the first quiet boundary.
check.py green (179). Babysit 21:16Z: box ×4 healthy @15.5–17.8k,
0.376–0.387 s/step — **A-s0 probe 9.4472@16.5k, first control under
9.5 and now below B's 9.59@17.5k** (the B-early-lead-was-transient
read strengthens); B aux-off 3.689@17.8k still at/below every
control's action loss (controls 3.77–3.91); draws run 2 @11.5k/25.8k
@100% util. No Discord traffic. GPUs busy + CPU queue non-empty
(#16 follow-on instruments: subset materializer + plan builder;
stage-2 sign pre-reg) → `run_work_next` armed per no-idle-pauses.*

*Previous update 2026-08-05 21:14Z (real `date -u`) — tick: **both chains
healthy, no Discord traffic; s1 watch item RESOLVED** (log had
advanced 15500→15720 — it was the probe/save boundary as suspected,
not a stall). Box ×4 @15.5–17.5k, 0.377–0.387 s/step, util 63–99%,
grad norms nominal: controls A-s0 total 4.12 @16.5k / s1 4.11
@15.7k / s2 4.23 @15.5k (action 3.73–3.91), **B aux-off total
3.685 @17.5k — still at/below every control's action loss** (margin
~0.05 vs s1's 3.73, continuing to narrow). Probe: **B 9.59@17.5k —
B's first sub-10 probe** (joins s2's 9.92@14.5k), trending toward
the <9@30k gate. Draws run 2 @11.4k/25.8k @100% util on the ~5 h
pacing. GPUs busy + CPU queue non-empty (#16 rig benchmark pre-reg
draft, stage-2 sign pre-reg) → `run_work_next` already armed per
no-idle-pauses.*

*Previous update 2026-08-05 21:12Z (real `date -u`) — work session: **IDEAS
#18.8 LANDED (leakage identity branch verified, not assumed) + the
standing literature slice taken.** #18.8
([journal](journal.md)): `bijou.eval.leakage`'s same-repo-id branch
now asserts episode-count equality vs the panel copy AND compares
per-episode length fingerprints (jsonl v2 / parquet v3; asymmetric
metadata fatal; same-dir shortcut) — a filtered-and-renumbered
corpus keeping its repo id can no longer certify a false PASS.
Mismatch = SystemExit demanding `source_provenance.json`. +4 tests
(179 green), `check.py` green; full-corpus identity cert re-run
PASSED (5267 radioactive / 47240 checked, 4.1 s); mutated-count
production copy fails loud. **Unblocks derived-corpus training
(#9, #13 repair).** Literature slice (~20 min, banked in ideas +
journal): **SnapFlow** (2604.05656) — self-distill flow VLAs to
1-NFE, no teacher, ~12 h/1 GPU, π0.5 1-step ≈ 10-step teacher,
SmolVLA-validated ⇒ ideas #12's distillation leg is now an
in-budget arm; **LoRA-π0** (2607.10172) — r=32 saturation, frozen
vision encoder degrades (external support for #11's
grounding-bottleneck read) ⇒ ft-protocol arm for #16. Babysits
21:04/21:10/21:11Z: box ×4 healthy @15.2–17.5k, 0.38–0.40 s/step
(**B aux-off total 3.69 @17.5k — still at/below every control's
action loss (3.69–3.79), margin narrowing**; watch item: s1 log
paused @15500 across two polls ~90 s apart, util fine — likely the
15.5k probe/save boundary, next tick verifies advance); draws run 2
@11.1k/25.8k on the ~5 h pacing. No Discord traffic. GPUs busy +
CPU queue non-empty (#16 rig benchmark pre-reg draft, stage-2 sign
pre-reg) → `run_work_next` armed per no-idle-pauses.*

*Previous update 2026-08-05 21:05Z (real `date -u`) — tick: **both chains
healthy, no Discord traffic** (the one new message was our own #18.2
post). Box ×4 @14.5–16.7k, 0.37–0.39 s/step (one benign 10.2 s
probe/save-boundary blip on s2): controls A-s0 total 4.22 @15.5k /
s1 4.23 @15k / s2 4.00 @14.5k (action 3.83–3.90), **B aux-off total
3.66 @16.7k — still below every control's action loss.** Probe:
**s2 9.92@14.5k — first arm under 10**, trending well toward the
<9@30k gate. Draws run 2 at 10.3k/25.8k @99% util on the ~5 h
pacing. GPUs busy + CPU queue non-empty (#16 rig benchmark pre-reg
draft, #18.8 leakage assert, stage-2 sign pre-reg, literature slice
due) → `run_work_next` armed per no-idle-pauses.*

*Previous update 2026-08-05 ~21:25Z-labeled (really ~20:55Z) — work session: **IDEAS
#18.2 (FLOW-NOISE STABLE-TRIPLE RESEED) LANDED BEHIND A FLAG, BREAK
PRE-REGISTERED** ([amendment](posts/2026-08-05-noise-reseed-prereg.md)).
The deep-dive's top finding fixed at the design level: `bijou.eval`
gains `--noise-key {index,stable}` — `stable` keys each frame's flow
noise to blake2b(repo_id, episode, frame) through a numpy SeedSequence
(128-bit keying, draw number as entropy: no torch 32-bit manual_seed
trap, no DRAW_SEED_STRIDE, no birthday collisions), making flow
numbers corpus-composition-invariant. Default stays `index`
(byte-identical to every banked anchor) until the pre-registered flip:
first anchor boundary after the box reads, one flow-80k panel re-bank,
decision band `6.6232 ± 3·max(0.045, empirical σ_draw from tonight's
draws chain)`, state-copy/AR bitwise-identity as hard controls.
Report JSON + banner now record `noise_key`; SmolVLA path threaded;
Q3 forced pass verified to share noise under both keyings. **Oracle:
AR-100k panel recomputed bit-exact through the edited path (12/12
cells d=0 incl. the 5.8026 anchor)**; 7 new unit tests (175 green),
`check.py` green. Babysits en route (~20:45Z, ~21:20Z): box ×4
healthy @14.5–16.5k, 0.38–0.40 s/step — **B aux-off total 3.80
@16.5k, still below every control's action loss** (3.83–3.88
@14.5–15.5k); draws run 2 at ~10k/25.8k @99% util on the ~5 h pacing.
No Discord traffic. GPUs busy + CPU queue non-empty (#16 rig
benchmark pre-reg draft, #18.8 leakage assert, stage-2 sign pre-reg)
→ `run_work_next` armed per no-idle-pauses.*

*Previous update 2026-08-05 20:38Z (real `date -u` — NB the previous entry's
"~20:55Z real clock" label was stamped ~20 min ahead of reality;
clock-label skepticism stays warranted) — tick: **both chains
healthy, no Discord traffic.** Box ×4 @12.5–14.5k, 0.37–0.42 s/step
(one benign 11.3 s save-boundary blip on s2), util 64–94%, ~71–74
GiB: controls A-s0 total 4.26 @13.5k / s1 4.28 @13k / s2 4.36
@12.5k (action 3.91–4.00), **B aux-off total 3.81 @14.5k — still
below every control's action loss.** Probes converged into one band:
A-s0 10.55@12.5k (10.99@13k), s1 10.69@12.5k, s2 10.18@12.5k, B
10.95@14k — B inside the control envelope, all trending toward the
<9@30k gate. Draws run 2 at 7.5k/25.8k @97% util on the ~5 h pacing.
GPUs busy + CPU queue non-empty (#18.2 reseed design, #16 rig
benchmark pre-reg draft, #18.8 leakage assert) → `run_work_next`
armed per no-idle-pauses.*

*Previous update 2026-08-05 ~20:55Z-labeled (really ~20:35Z) — work session: **IDEAS
#18.1 (INSTRUMENT HARDENING PASS) LANDED**
([post](posts/2026-08-05-hardening-pass.md)). Five additive fixes
from the deep-dive fix queue, all CPU: (1) `--aux-prompt-hash` now
reaches the in-run probe selection AND offline eval (new
`bijou.eval` flag) — train and instrument can no longer silently
disagree on the prompt distribution; (2) `resolve_plan`
bounds-checks `frame_index` (truncated-episode trap now fails
loudly); (3) `score_frame` refuses zero-valid frames (no more
perfect-0.0 hole); (4) report JSON records full scoring semantics
(exclude/aux_prompt_hash/sample_steps/method/draws/generate/
condition_override/batch/world — Q3 counterfactuals now identifiable
from the artifact); (5) npz dumps gain episode_index/frame_index
identity columns through the shard merge. **Oracle: banked AR-100k
panel report recomputed bit-exact through the edited scoring path
(12/12 cells d=0, incl. the 5.8026 anchor)**; 3 new unit tests, 168
total, check.py green. Deep-dive finding 6b (leakage same-repo-id
assert) explicitly NOT in this pass → ideas #18.8. Babysits en
route (20:26Z, 20:33Z): box ×4 healthy @12.0–14.0k, 0.37–0.42
s/step — **B aux-off total 3.80 @14k, still below every control's
action loss** (3.91–3.98 @12.4–13.2k); A-s0 probe 10.55@12.5k (next
gate <9@30k); draws run 2 at 7.1k/25.8k @99% util on the ~5 h
pacing. No Discord traffic. GPUs busy + CPU queue non-empty (#18.2
Q3/reseed design, #16 benchmark pre-reg draft) → `run_work_next`
armed per no-idle-pauses.*

*Previous update 2026-08-05 20:26Z (real clock) — tick: **both chains
healthy, no Discord traffic.** Box ×4 @11.7–13.5k, 0.38–0.42 s/step,
util 56–100% sampling: controls A-s0 total 4.26 / s1 4.31 / s2 4.26
(action 3.93–3.99), **B aux-off total 3.87 @13.5k — still below
every control's action loss**; A-s0 probe 10.55@12.5k (next gate
<9@30k), grad norms nominal. Draws run 2 at 6.3k/25.8k @99% util,
~24% in ~1.1 h — consistent with the ~5 h pacing, ETA ~00:1xZ.
GPUs busy + CPU queue non-empty (ideas #18 cheap hardening pass) →
`run_work_next` armed per no-idle-pauses.*

*Previous update ~20:30Z — work session: **IDEA #2a
(LENGTH-BUCKETED BATCHING) LANDED — and the sim says DON'T spend a
GPU screen on it under the current recipe.**
[Post](posts/2026-08-05-bucketing-impl-sim.md).
`--bucket-by-length` in `bijou.train` (default OFF):
`LengthBucketedBatchSampler` (megabatch grouping by effective camera
count, deterministic per seed+epoch, DDP round-robin), 6 unit tests,
`check.py` green, **all three CPU loss oracles bit-exact** with the
flag off (2.7903/1.9152, 4.9232/4.8631, 27.8262/27.7701), gradflow
probe green, CPU smoke with flag ON works. **Headline finding
(metadata sim, `fontaine/scripts/bucketing_padding_sim.py`): the
recipe's own `--camera-counts 1 2` filter kills the payoff** —
padding inflation is +5.09% → ceiling ~3.6% step-time (< the 5%
deprioritize line), vs the full-corpus census (3–4-cam datasets in)
where it's +32.55% → −23.8% padded tokens ≈ 19% ceiling. Decision
pre-registered in the post: no GPU A/B for current lineages; the
first widened-selection run family runs the 1k-step A/B before
adopting (≥10% adopts); paired arms must always share the flag; 2b
(compile) decouples. Ideas #2 → `screening`. **Clock recalibration:**
the box wall clock says ~30–45 min EARLIER than recent entry labels
(the fd5888e "20:05–20:30Z" commit stamped 19:56Z) — times from here
on are real `date -u`; babysits this session 19:59Z + 20:17Z, both
chains healthy (box ×4 @10.0–12.9k, 0.37–0.40 s/step, **s1 probe
11.01@10k — the last <12@10k gate PASSED, placeholder below fixed**;
B total 3.93@12.9k still below every control's action loss; draws
run 2 @5.5k/25.8k, 100% util). No Discord traffic.*

*Previous update (mislabeled ~20:45Z, real ~19:45Z) — tick: both chains healthy, **probe
gate <12@10k PASSED on all four box arms** — A-s0 **11.71@10k**,
s1 **11.01@10k** (was the watch item at 12.64@9k — dropped to
11.82@9.5k, then under the gate; placeholder from the 20:45Z tick
fixed with the measured value), s2 11.30@9.5k, B aux-off 11.64@11k. **B's early probe
lead is GONE**: it now sits inside the control envelope
(11.3–11.8) — the E3 @2.5k offset (16.9 vs 24.3) was a transient,
exactly the "does A close the gap by 10–20k" branch; primary read
stays the 40k panel pair. B's total loss 3.94@11k still below every
control's action loss (4.05–4.12@10k). Pace 0.38 s/step ×4 (one
benign 10.3 s blip on B at a save boundary). Draws run 2 at
3.2k/25.8k @99% util, on the ~5 h pacing. No Discord traffic. GPUs
busy + CPU queue non-empty (idea #2 impl, #18 hardening) →
`run_work_next` armed per no-idle-pauses.*

*Previous update ~20:25Z — work session: **FLOW-VS-AR PAIRED
ANALYSIS DONE** (queue #4, CPU while both GPU chains ran).
[Post](posts/2026-08-05-flow-vs-ar-paired.md); script
`fontaine/scripts/flow_vs_ar_paired.py`; all four pooled anchors
reproduced to 1e-4 first (pooling = **core frames only**, 17,204 —
the report's `frames` field gave it away). **Headline: the 0.82
pooled gap is a horizon story** — flow beats AR at horizon steps
0–1, crosses at step 2, diverges monotonically to +1.2 by step 40.
Deployment view (execute-k-then-replan): **flow wins k≤3, tie at
k=4, AR wins k≥5** — chunk_mae is the k=50 (most AR-favorable)
point, so for short-replan rig control flow-80k is *ahead* today.
Cuts: flow win rate 36.5% of frames; deficit grows with motion
(+0.59 still → +0.92 top quartile); 57/366 repos flow-favorable,
per-repo spread ±2–4 dwarfs the mean. Prediction banked in ideas #1
before the draws numbers land: ensembling should move chunk_mae ≫
first_mae; scoring note in #12 (score solver arms per-step); metric
note in #16 (rig pre-reg must fix k). Babysits en route: box
healthy ×4 @9.5–10.2k (probe convergence — B's early lead is gone:
A-s0 11.77@9.5k / s2 11.86@8.5k / B 11.83@10k; s1 12.64@9k
trending down, watch vs the <12@10k gate; B aux-off total 3.99–4.08
@10.2k still below every control's action loss); draws run 2 at
~2.1k/25.8k frames, util sampling 59–95% healthy.*
(owner mandate 17:50–18:01Z; resumed from the 429-killed draft).
Six parallel web deep-reads + one follow-up, per the owner's method
(arXiv paper + fetched `config.json` per candidate, post-cutoff
epistemics). [Post](posts/2026-08-05-trunk-survey.md); ranked queue
mirrored into ideas #17. **Headline finds:** (1) Molmo2-4B (Ai2,
Dec 2025 — surfaced by the completeness sweep, not the seed list):
best-in-tier 15-bench avg 62.8 vs Qwen3-VL-4B 58.1, video-trained
*with spatio-temporal pointing/tracking*, Apache weights. (2)
**Molmo2-4B, InternVL3.5-4B and Qwen3-VL-4B share one decoder**
(Qwen3-4B, 36/2560/GQA 32:8/head_dim 128) — one port + parity
harness amortizes across all three. (3) InternVL3.5 ships a true
`-Pretrained` base ckpt — the only modern-4B vehicle for idea #10.
(4) V-JEPA **2.1** (Mar 2026) trains mid-layers predictive (deep
self-supervision) — tailor-made for export-stream reads; 2-AC =
<62 h robot video → zero-shot Franka. (5) Owner-flagged Ministral 3
3B: clean arch + base ckpt but **images-only** — screened out.
Verdict: E4B rung first (zero cost), then Molmo2-4B, then
InternVL3.5-4B (base-vs-IT), V-JEPA 2.1 arm in parallel;
Qwen3-VL-4B reserve. No Qwen3.5-VL exists (checked). Babysits
en route: box healthy ×4 @8.0–9.1k (B aux-off 4.043 @9.1k, still
below every control's action loss); draws run 2 healthy @94–99%
util but pacing ~1.4 frames/s ⇒ **~5 h for the draws-10 run, not
~1.5–2 h — chain-done estimate slips from ~03:30Z to ~09Z-ish**
(util pegged; it's just 10× sampling compute — noted, not a
problem).*

*Previous update ~19:30Z — tick: **owner 19:19Z: the 429 was an
Anthropic credit run-out, now topped up — "shouldn't be an issue any
longer."** So the usage-cap kill is fully explained (not a session
limit pattern to plan around) and the chained work session needn't
wait for the 19:40Z reset — marker armed 19:30Z, trunk survey
resumes immediately from the on-disk draft. Both chains healthy:
box ×4 @7.9–9.0k, 0.38 s/step, controls 4.47–4.62 (action
4.13–4.19), **B aux-off 4.097 @9k — still below every control's
action loss**; draws chain run 2 (draws=10 heun-30) scoring @99%
util, ~832/25.8k frames. Acked in-channel.*

*Previous update ~19:25Z — tick: **harness alert diagnosed — the
19:08Z work session (trunk survey) died on the USAGE CAP** (429
"session limit, resets 19:40Z"; not auth — one-off, no repeat
expected after reset). Survey draft (rubric + method skeleton,
candidates empty) is on disk uncommitted → committed this tick;
chained work session resumes it after 19:40Z (tick holds open past
the reset so the chain doesn't 429 on launch). **Draws chain E1
gate PASSED**: run 1 (N=1 heun-30) chunk_mae **6.624** vs owner box
6.6232 (Δ0.001, band ±0.03), first_mae 1.933 ≡ owner's 1.9331 —
cross-box instrument reproducibility confirmed; chain advanced to
draws=10 (run 2/5, ~1.5–2 h each, chain done ~03:30Z). Box healthy
×4 @7.0–8.3k, 0.38 s/step: controls 4.60–4.67 (action 4.23–4.24),
**B aux-off 4.169 total — still below every control's action loss
at 8k**. Posted in-channel.*

*Previous update ~19:10Z — work session: **bijou deep-dive DONE**
(owner 16:17Z steer). All 57 files / 22.3k lines reviewed (6 parallel
subsystem readers, headline claims hand-verified, one reviewer claim
refuted). **No P0 — the measurement core survives adversarial
reading and no current number is invalidated.** Deliverable:
[ranked findings post](posts/2026-08-05-bijou-deep-dive.md); fix
queue = ideas.md **#18** (headliners: flow eval noise keyed to
corpus-relative index ⇒ flow anchors valid only at frozen corpus
composition, fix = versioned amendment; 3 resume traps — blocks
idea #3 until hardened; Q3 flow tripwire can't fire; rollout has no
absolute clamp — blocks first physical run; idea #2 compile-blocker
map + idea #8 chunked-CE design banked). Runs @19:04Z: box healthy
×4 (B aux-off 4.14 @7.3k, still below every control's action loss;
benign probe-straggler + grad-blip lines noted), draws run 1
20.2k/25.8k @99% — E1 number ~19:35Z, tick watches. No Discord
traffic.*

*Previous update ~18:47Z (work session: **charter v1.1 — the
owner-steered rules pass is DONE** — eight steering deltas codified
into charter + prompts ([journal](journal.md), charter §11 amendment
log); `check.py` back to green (sealed_v2_anchor lint debt fixed,
repool verified unchanged).)*

## ⚡ The second box (192.222.55.210) — batch RUNNING

Pre-reg: [box batch](posts/2026-08-05-prereg-box-batch-4xh100.md)
(commit cc0b922, posted before launch). Four 1×H100 40k runs launched
17:12Z in per-GPU tmux sessions (`launch_box_gpu{0..3}_*`):

| GPU | run | seed | tmux / log |
|-----|-----|------|------------|
| 0 | A-s0 control | 0 | `~/train_fontaine_arb_rcond_40k_1xh100.log` |
| 1 | B-s0 aux-off | 0 | `~/train_fontaine_arb_rcond_auxoff_40k_1xh100.log` |
| 2 | A-s1 control | 1 | `..._s1.log` |
| 3 | A-s2 control | 2 | `..._s2.log` |

- **E1 hard gate PASSED on all four** (17:15Z): 878 datasets /
  38,571 train + 4,301 holdout = 42,872 episodes / dims 6/6 / 103
  dropped — identical, and B-s0's log carries **no aux line** while
  A's shows fields + weight 0.5. Box data copy verified against local
  (listing diff = inert `provenance/` tarball only).
- **E2 first-poll PASSED (17:18Z, util rule):** all four stepping at
  0.43–0.54 s/step (band 0.4–0.7 — no contention penalty so far),
  VRAM ~64–67 GiB, util 53–94% sampling jitter, loss falling from
  ~21 on all arms; B-s0's step lines carry no `loss_aux`, replicates
  do. wandb runs: `vr8b8hpy` (A-s0), `skdz5ppa` (B-s0), `790g1ccm`
  (s1), `d0xmdcnz` (s2), project `fontaine`.
- Each GPU chains its panel eval (k4l2, `--dump-predictions`) after
  40k. ~5–6.5 h train + ~1.7 h eval ⇒ all reads by ~02Z.
- **Babysit every ~30 min of session time**: liveness + s/step
  (0.4–0.7 healthy, >0.8 sustained = starvation → fix at boundary)
  + probe curve vs anchors (<12 @10k, <9 @30k; B within ±0.3 of A).
  Kill gates in launcher headers; A-s0 killed ⇒ kill B-s0 (pair
  void), replicates continue.
- **18:05Z babysit: healthy ×4** (steps 2.5–3k, 0.37–0.39 s/step,
  util 68–93%, ~70 GiB each; losses ~21 → 5.2–5.4). **E3 already
  broken at 2.5k, in B's favor**: probe B-s0 16.85 vs A-s0 24.32
  (matched step; B 15.53 @3k) — aux-off descends much faster early.
  No kill gate tripped; primary read stays the 40k panel pair.
  Surprise logged ([journal](journal.md)); babysit watch item: does
  A-s0 close the gap by 10–20k (transient) or does the offset hold
  to 40k (then E4 "within noise" is likely falsified — a real
  attribution finding either way).
- **18:49Z tick: healthy ×4** (steps 5.0–6.0k, 0.37–0.41 s/step,
  util 68–74%, ~70 GiB). Losses: controls 4.80–4.87 (action
  4.39–4.44), **B aux-off 4.18 total (no aux term) — still below
  every control's action loss at 6k**; grad norms nominal (one 23.4
  blip on s1, loss unaffected). No kill gates near. Draws chain run
  1 alive at 11k/25.8k, 100% util — E1 number expected ~19:20Z.
- **18:12Z tick: healthy ×4** (steps 2.5–3.5k, 0.38 s/step, util
  65–83%). **Matched-2500 probe now complete across all four**:
  controls A-s0 24.32 / s1 29.72 / s2 29.69 (seed envelope
  [24.3, 29.7] — early probes are noisy, ±0.3 band was optimistic
  for early steps), **B-s0 16.85 — ~7.5 below the *best* control**,
  well outside the seed envelope. The E3 early aux-off lead survives
  the noise-floor check.
- **rsync-back live**: local tmux `fontaine-rsync`
  (`~/boxsync_loop.sh`, 20-min cadence): logs + eval reports + latest
  two saves per run → `~/boxsync/`.
- **Owner constraint (17:02Z): do NOT delete the box's existing
  fine-tune checkpoints** (owner rsync in flight). No cleanup of any
  kind runs on that box.
- Code on box: branch `fontaine` @ cc0b922 (pushed over direct SSH;
  box `.venv` reused — torch 2.11.0+cu130 both boxes, no seam).

## What the LOCAL GPU is doing: noise-draw chain (launched 18:25Z)

**Sealed baseline DONE 18:24Z** — anchors banked (next section).
Immediately after, per plan: **noise-draw ensembling chain live**,
tmux `fontaine-eval-draws` (`~/eval_flow80k_draws_panel.sh`, 5 runs
≈ 9 h → done ~03:30Z). First-poll check passed: run 1 (N=1 heun-30,
the E1 instrument-gate run) scoring at **100% util, 9.2 GiB**. The
launcher itself stops the chain if E1 fails (N=1 must reproduce
6.6232 ±0.03 — owner's 12:20Z box eval). Per-run logs
`~/eval__bijou_flow_artrunk...draws{N}_{solver}.log`. Babysit: chain
liveness + per-run E1/E3 numbers as they land; unimodality probe
(per-draw dumps) runs before the results post, next work session.
- **19:20Z: E1 GATE PASSED** — run 1 chunk_mae **6.624** (owner box
  6.6232, Δ0.001 ≪ ±0.03 band), first_mae 1.933; state-copy 11.785;
  Q3 condition sensitivity 0.898 over 5,070 labeled non-success
  frames. Report + html in local `reports/`. Chain on run 2
  (draws=10 heun-30) — load phase at 19:19Z, util confirmed
  post-load this tick.

## Sealed-panel anchors — BANKED 18:24Z (posted in-channel)

From `reports/eval__bijou_arb_rcond_100k_ddp4__step_100000__panel_curated_v0_k4l2_sealed.json`
(25.8k scored frame-policies, 17,204 pooled frames/policy):

| policy | v1 (as drawn) | v2 (census repos removed) |
|---|---|---|
| bijou@100k | **5.7540** | **5.6903** (±5e-3 method) |
| bijou@100k+fields | 5.7482 | 5.6962 (±3e-3) |
| state-copy | 11.6635 | 11.5883 (±4e-2) |

- v1 in band: expectation was 5.8017 ±0.15 → gap −0.048 ✅;
  state-copy −0.12 vs the primary draw (two draws agree well).
- `+fields` indistinguishable from bare bijou (−0.006) — consistent
  with the mainline "aux within noise at the endpoint" read.
- v1→v2 shift ≈ −0.07, matching the amendment's prediction; method
  error ~15× smaller than the shift
  ([amendment](posts/2026-08-05-sealed-plan-v2.md)).

## Banked this session (no GPU needed): 80k flow panel number

Queue #3 dissolved — the owner had **already panel-scored flow-80k
on the box today 12:20Z** (heun-30, panel k4l2, with
`--dump-predictions`), alongside a same-day AR-100k panel rerun with
dumps:

- **flow-80k @ heun-30: chunk_mae 6.6232, first_mae 1.9331**
- **AR-100k: chunk_mae 5.8026 (anchor, bitwise), first_mae 2.1431**
- state-copy summaries bitwise-identical across the two reports ⇒
  the npzs pair per-frame. Flow still trails AR by 0.82 pooled but
  **beats it on first_mae** (1.93 vs 2.14, the grounding-sensitive
  column).

All eight files pulled to local `reports/` (17:14Z). Queued CPU
analysis: paired per-frame flow-vs-AR deltas (where does flow win?)
— feeds a results post + the solver/ensembling ideas (#1, #12).

## Work session ~18:45–19:05Z — the rules pass (charter v1.1)

One bounded item per the owner's order (18:36Z: "let's start with
the rules pass"): reviewed charter + all prompts against the day's
accumulated steering; eight deltas codified (charter §11 amendment
log; [journal](journal.md) narrative): §0 north star + startup
velocity, §1 loaned compute, §2 measure-versioning + rig-instrument
clarification, §3 first-poll util + **no-idle-pauses standing rule**,
§6 post-cutoff epistemics, §9 chaining semantics + Discord house
style; `tick.md`/`work.md` updated to chain work whenever GPUs are
busy and CPU items are queued. `check.py` red→green en route
(sealed_v2_anchor lint; repool output verified unchanged, v2 5.6903
reproduces). Both run chains re-checked twice (18:40Z, 19:00Z),
healthy. Marker armed → bijou deep-dive chains next.

## Earlier work session (17:03Z→) — what happened

1. Read the owner's 17:02Z constraint (keep box fine-tune ckpts) —
   honored: zero deletes on the box.
2. Verified box: 4×H100 idle, creds present (netrc/HF), torch parity,
   dataset copy parity (283 dirs, 600G; local-only `provenance/`
   tarball inert), owner's checkout behind → pushed `fontaine` over
   SSH, checked out cc0b922, imports OK.
3. Wrote + posted the [batch pre-reg](posts/2026-08-05-prereg-box-batch-4xh100.md)
   (execution supersedes the local sequential plan; science of the
   [paired pre-reg](posts/2026-08-05-prereg-paired-auxoff-40k.md)
   unchanged; new E5 = seed-noise floor with pre-registered decision
   rule). Banner added to the paired pre-reg. `check.py` green.
4. Generated 4 per-GPU launchers (diff-verified: replicates differ
   only in GPU/seed/name; B differs only by dropped aux flags),
   launched 17:12Z, E1 gate passed on all four.
5. Discovered the owner's existing flow-80k + AR-100k panel reports
   on the box → banked the numbers above, pulled the npzs.
6. rsync-back loop started (`fontaine-rsync` tmux).

## Bootstrap scoreboard (charter §10)

- §10.1–§10.6 — **done** (sealed anchor banked 18:24Z: v1 5.7540 /
  v2 5.6903).
- §10.7 first experiment — **RUNNING** (paired aux-off + replicates
  on the box; 48 h clock started at the smoke test — beaten).

## Owner steering log (active items)

- **21:43Z (conversational, replied 21:45Z, exchange live): MAJOR
  REWEIGHT — #16 rig-benchmark execution PARKED, short-term focus =
  comm-holdout MAE + limit attribution.** Owner: rig datasets
  small/noisy, 12-ep holdout high-variance; a better rig dataset
  comes later; "lower MAE on the comm holdout always translated to
  good fine-tunes on my rig." Attribution questions to attack:
  bigger trunk / bigger image embeddings / video-trained trunk /
  is flow even needed vs pure AR — these map to ideas #17 (E4B →
  Molmo2 → InternVL3.5, V-JEPA 2.1), #11 (grounding; owner's
  failure anecdote is gripper *placement*, i.e. grounding), #12/#1
  (flow-vs-AR + ensembling). Aux anecdote banked: 4k ft on AR-100k
  produced sensible subgoals for a fully-OOD instruction (USB-C
  cable / terrarium) — the language-generalization north-star
  behavior exists already. Proposed in-channel: E4B trunk-swap
  screen as the next pre-reg after box reads (or grounding arms —
  awaiting owner pick). #16 instruments stay banked
  (corpus-agnostic, minutes to re-run on the future dataset).

- 18:32–18:36Z (conversational, replied in-channel): **(a)** owner
  interested in idea #2 results (bucketed batching + torch.compile
  prefix) — status given (impl any work session, A/B needs a quiet
  GPU boundary ⇒ after box reads land); **(b) keep review order,
  rules pass first** (confirmed); **(c) STANDING RULE: no idle
  pauses while GPUs are busy** — owner: "we should be able to do a
  lot of work items while the GPUs are busy… unnecessary pauses
  right now." Adopted: GPU-busy windows = CPU work-item windows;
  `run_work_next` touched 18:38Z, work session chains immediately
  (order: rules pass → bijou deep-dive → trunk survey → flow-vs-AR
  analysis → idea #2 impl). Save to memory.
- 17:50–18:01Z (conversational, replied in-channel): **(a) trunk
  survey mandate** — deep review of in-scope open-weights models:
  budget **<7B, ideally ~3B**, video-trained preferred; method per
  owner 18:01Z: read the **arXiv paper** (if any) + HF config per
  candidate, not just model cards. Multi-turn = later-stage research
  area (noted, not started). → queued in the owner-steered reviews
  block (item 5c). **(b) Ministral 3 3B** flagged by owner —
  first-read posted (3.4B LM + 0.4B vision enc, 256k ctx, Apache
  2.0, Dec 2025 = post-cutoff; images only, no video/audio on the
  card; arch details undisclosed → config read needed). Candidate on
  size/license; misses the video-trained preference. **(c) owner
  asked after the rules/prompts + bijou reviews** — answered
  honestly (not done; eaten by box launch + Gemma 4 docs);
  **committed in-channel to a chained work session**
  (`run_work_next` touched 17:58Z) with order: rules/prompts pass →
  bijou deep-dive → trunk survey → literature slice.
- 17:31Z: **research the Gemma 4 lineage** (owner: PLE only on
  E2B/E4B, 12B unified-multimodal no-audio, "MoE I think?"; read
  the HF blog) → **DONE this tick**: blog read, `docs/gemma4.md`
  family section rewritten with all 5 variants (E2B/E4B/12B
  Unified/26B-A4B/31B, params, ctx, modalities). Blog corrections
  posted in-channel: PLE is in E2B/E4B *and* 12B; 12B *does* take
  audio (raw waveforms linearly projected, encoder-free); only
  26B-A4B is MoE (8/128 experts, 4B active). Summary posted 17:41Z.
- 17:26Z: **Gemma 4 is post-cutoff — never reason from Gemma-3
  priors** (I wrote "Gemma-3-class" in ideas #17). → **DONE this
  tick**: `docs/gemma4.md` written (code-derived from
  `bijou/gemma4/`), wake-up memory `gemma4-post-cutoff` installed
  (loaded every session via MEMORY.md), ideas #17 line fixed to
  "larger Gemma-4 variants (E4B/12B)". Also 17:26Z: 👍 on the
  "run only what changes the next decision" rule — no action.
- 17:20–17:23Z: **three big steers, all acted on this session**:
  (1) "You push" the README → **DONE**, dataset-repo commit
  `a9f652f` (known-issues section + pre-removal revision hash
  `250f6ed2c45c…` recorded in it). (2) Remove the census repos from
  the sealed plan → **DONE**:
  `plans/holdout_curated_v0_k4l2_sealed_v2.json` (core −52 frames /
  13 eps, labeled −26; [amendment posted](posts/2026-08-05-sealed-plan-v2.md);
  v1 deprecated; v2 anchor re-pools from the v1 report's per-dataset
  means when the running eval lands — note: sealed run has NO npz
  dump; the recompute (`fontaine/scripts/sealed_v2_anchor.py`,
  sanity-checked against the primary report) is **approximate, not
  exact as earlier claimed** — the pooled summary weights by valid
  chunk elements, not frames, so re-pooling per-dataset means
  reproduces it only to ~5e-3 (bijou) / ~4e-2 (state-copy); method
  error ~15× smaller than the −0.07 v1→v2 shift, negligible vs the
  0.15 band, quoted with the anchor).
  (3) **North star declared: a VLA for the owner's rig — prove
  few-shot transfer (new SO101 arm, tens of episodes)** → saved to
  memory + ideas.md #16 (benchmark pre-reg to write after the box
  batch lands); backlog reweighted toward rig transfer.
- 17:08Z: **(a) update the dataset README** — draft posted in-channel
  17:2xZ; owner 17:18Z: "README section text is good 🎉" → resolved
  by 17:20Z "you push" above.
  **(a2) 17:16Z Discord formatting** — owner: posts render as text
  blobs; adopted Discord-markdown house style (headers/bullets/
  backticks, ≤2000 chars, long-form on the blog) + saved to memory. **(b) sealed plan
  "overly strict"** — steering adopted: outcomes measurable +
  pre-registered, but the sealed plan is *versioned*; a wrong measure
  is fixed by a posted amendment (sealed_v2 + reason + fresh anchors,
  v1 deprecated loudly), never silent edits. Codify in the rules pass
  (queued next session). Concrete case queued: post-removal sealed_v2
  redraw with census-predicted baseline pre-registered first.
- 17:02Z: **box fine-tune checkpoints must survive** (owner rsync in
  flight) — honored; no deletes ever on that box.
- 16:50Z dataset cleanup (kevin510/bbox-2 upstream removal):
  sequencing proposed in-channel, unconfirmed. **Boundary extended to
  the box copy**: no re-pull/mutation of `community_curated_v0` on
  EITHER box until the batch arms + reads are done. Record the
  pre-removal HF revision hash before any upstream push lands.
- 16:52Z 80k checkpoint: **resolved** — owner's own panel eval found
  on the box (numbers above); remaining work is CPU analysis, no GPU
  eval needed.
- 16:21Z rules/prompts review: **DONE ~19:00Z (charter v1.0 → v1.1)**
  — amendment list in charter §11, narrative in [journal](journal.md);
  prompts (`tick.md`/`work.md`) updated to the no-idle-pauses chain.
- 16:17Z bijou code deep-dive: **DONE ~19:10Z** —
  [ranked post](posts/2026-08-05-bijou-deep-dive.md); no P0, fix
  queue in ideas #18.
- 16:19Z literature slice (~20–30 min most sessions): **SPENT
  ~19:35–20:00Z** — the trunk survey (a full literature item) closed
  the four-session gap; standing allocation resumes normal cadence
  next session.

## Queue (depth 5)

1. **Babysit the box batch + the local draws chain** (every ~30 min
   session time). Box: see box section. Draws chain: liveness +
   E1 gate result on run 1 (~20:00Z), then per-run numbers. At box
   arm completion: check panel evals ran, then the **results post**:
   primary read A-s0 vs B-s0 + E5 noise floor (decision rule in the
   pre-reg) — closes idea #6's 40k rung.
2. ~~Sealed anchor~~ **DONE 18:24Z** — banked + posted (section
   above).
3. ~~Noise-draw chain launch~~ **RUNNING** (launched 18:25Z; see
   local-GPU section). Remaining: unimodality probe before the
   results post.
4. ~~Paired flow-vs-AR per-frame analysis~~ **DONE ~20:25Z** —
   [post](posts/2026-08-05-flow-vs-ar-paired.md); horizon-crossover
   finding; predictions banked into ideas #1/#12/#16.
5. **Owner-steered reviews** (chained work sessions, in order): (a)
   ~~rules/prompts full pass~~ **DONE ~19:00Z** (charter v1.1), (b)
   ~~bijou deep-dive~~ **DONE ~19:10Z**
   ([ranked post](posts/2026-08-05-bijou-deep-dive.md); fix queue =
   ideas #18), (c) ~~trunk survey~~ **DONE ~20:00Z**
   ([post](posts/2026-08-05-trunk-survey.md); ranked queue in ideas
   #17: E4B → Molmo2-4B → InternVL3.5-4B → V-JEPA 2.1 arm;
   Qwen3-VL-4B reserve; the E4B screen pre-reg is the natural next
   queue-refill item once box reads land), (d) ~~flow-vs-AR
   per-frame analysis~~ **DONE ~20:25Z** (queue #4,
   [post](posts/2026-08-05-flow-vs-ar-paired.md)), (e) ~~idea #2a
   bucketing implementation~~ **DONE ~20:30Z**
   ([post](posts/2026-08-05-bucketing-impl-sim.md); GPU screen
   pre-registered CONDITIONALLY — sub-threshold under the current
   recipe, sim banked instead). Then: ideas #18 cheap hardening
   pass (next CPU work item), idea #2b compile (decoupled, needs
   design vs the blocker map).
6. Stage-2 sign-convention pre-reg draft (mirror trio) — backlog.
8. **Ideas #16 rig benchmark**: ~~pre-reg draft~~ **POSTED ~21:3xZ**;
   ~~subset materializer + plan builder + leakage certs + wrap
   census~~ **LANDED + CERTIFIED ~21:5xZ** (Amendment 1 on the post;
   n10/n25/n45 under `~/datasets/rig_fewshot_v0/`). **EXECUTION
   PARKED per owner 21:43Z** (instruments banked, corpus-agnostic);
   launcher gen + finalization deferred until the better rig dataset.
9. **Comm-MAE limit-attribution front (owner 21:43Z)** — ~~E4B
   screen pre-reg~~ **POSTED ~22:4xZ**
   ([post](posts/2026-08-05-prereg-e4b-screen.md)). The freed 4×H100
   at ~02Z goes here. ~~Chunked-backward impl + oracles~~ **LANDED
   ~23:0xZ unconditionally** (`--backward-chunks`; Amendment 1 on the
   pre-reg corrects the chunk-mean sketch — global-count
   normalization, exact for unequal token counts; check.py 191
   green): an OOM at the smoke now costs zero launch delay.
   Remaining before launch: results post for the box batch, e4b ckpt
   download, parity spot-check, B12 memory smoke, finalization
   amendment (σ_seed from the E5 reads + ladder rung).
   ~~`read4_energy_score`~~ **LANDED 23:0xZ-labeled session (real
   ~22:15Z)** — all four fairness reads execution-ready.
7. **Ideas #18 instrument hardening**: ~~the cheap pass (#18.1)~~
   **DONE ~20:55Z** ([post](posts/2026-08-05-hardening-pass.md);
   oracle bit-exact, check.py green). Remaining GPU-busy CPU items:
   #18.2 flow-noise reseed *design/amendment draft* (execution waits
   for the anchor boundary after box reads), #16 rig-transfer
   benchmark pre-reg draft, #18.8 leakage 6b assert, stage-2
   sign-convention pre-reg (item 6).

## Handoff notes for the tick loop

Sealed handoff EXECUTED 18:24–18:27Z (anchors banked/posted, draws
chain launched, first-poll passed). Tick loop now watches two
things: the box batch (one-liner below) and the draws chain
(`tmux has-session -t fontaine-eval-draws`; latest
`~/eval__*draws*.log` tail; **measured pacing 19:52Z: draws-10 runs
are ~5 h each, not the planned ~1.5–2 h** — chain-done ~09Z-ish; a
long-running run 2 is healthy, don't diagnose. Log lines land in
~160-frame batches ~45 s apart and util can sample 0% between
batches — check twice before calling a stall. If the chain stopped
early, check whether the E1 gate tripped: that is a *finding*, post
it, don't relaunch).

Box babysit one-liner (tick or work):
`ssh ubuntu@192.222.55.210 'tail -2 ~/train_fontaine_*.log; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader'`

Known safe-to-ignore: `wandb/` untracked at repo root (smoke
scratch); owner tmux sessions on the box (`5`, `rigjudge`,
`watchdog`) — theirs, do not touch.

Usage-cap note (19:12Z alert; RESOLVED 19:19Z — owner: Anthropic
credits ran out, topped up, "shouldn't be an issue any longer"):
429s can kill a session mid-work (`terminal_reason: api_error, 429`, reset time in
the alert/log tail). Diagnosis path: tail the named harness log,
look at the last `result` JSON. Uncommitted work survives on disk —
commit it in the next session. If a chain marker is armed just
before a reset boundary, prefer holding the live session past the
reset so the chained session doesn't die on launch.

## Utilization footer

Trailing-7-day GPU-hours on experiments / total: local **~24.1 / ~24.4**,
box **~42.9 / ~42.9** (as of 23:3xZ: box — masked q4 reliance eval
COMPLETE ~19:05Z ≈ 0.5 h; the rung 4→8 memory-ladder smokes
19:3x–22:5xZ ≈ 3 GPU-h (four OOM rungs died in minutes each; rung 7
trained to its verdict; rung 8 smoke green); **molmo2 AR 40k LIVE
since 22:57Z on all 4 GPUs** ≈ 2.2 GPU-h so far at 23:3xZ, step
540/40k, 2.19 s/step → ~24 h to 40k. Local — untrained-gen probe
≈ 0.1 h; otherwise idle-by-design since 18:1xZ; **next local
consumer pre-registered: A-s0 AR draws10_t1** — launch next chained
session. Explore/exploit: this session all-CPU exploit on the
GPU-busy window's queued instrument (ideas #19, owner ask); lit
slice skipped — instrument critical path, 16:04Z slice balance
carries.)
Stale detail below is the 18:1xZ snapshot:
(as of 18:1xZ: local — SnapFlow ftrig fine-tune
17:02→17:50Z ≈ 0.8 h COMPLETE at 4k + chained after-reads (rig draws
1/10 + panel-v2 guard) ≈ 0.6 h ending ~18:1xZ; box — arm C 40k
COMPLETE 16:02Z, its chained panel eval on GPU 0 live since 16:05Z
≈ 2.2 h @21,472/25,800, masked eval next → boundary ~19:0x–19:3xZ;
GPUs 1–3 idle pending the arm A launch call (owner rec posted:
arm A tonight, Molmo2 AR 4×DDP takes the box tomorrow). CPU-side this
session was the Molmo2 port sprint: WP1+WP2+full HF parity in one
session, all CPU — the no-idle-pauses rule at its best.)
Stale detail below is the 15:2xZ snapshot:
(as of 15:2xZ: sealed eval 1.9 h; noise-draw chain 18:25Z→04:12Z ≈
9.8 h COMPLETE; state probe ≈ 1.4 h; fairness probe ≈ 1.2 h; #18.2
flip re-bank ≈ 0.8 h ADOPTED; **SnapFlow distill 08:43→13:14Z ≈
4.5 h COMPLETE at 30k**; **SnapFlow endpoint-eval arc COMPLETE
13:14–15:10Z ≈ 1.8 h** — draws1 5.6036/1.7039, draws10
5.3675/1.5927, draws5 5.3918/1.6056, npz addendum 14:43–15:10Z —
frozen verdict PARITY-ADOPT published; **local GPU idle-by-design
since 15:10Z** — next local GPU work only via a new pre-reg),
box **~34.9 / ~34.9 GPU-h**
(4 arms trained ≈ 17 GPU-h + 4 chained panel evals ≈ 10 GPU-h; E4B
memory smoke ≈ 0.8 GPU-h NO-LAUNCH; **arm C state-dropout live since
08:10Z on GPU 0** @37,160/40k at 15:38Z (≈7.5 h so far), 0.374–0.39
s/step, in-run probe DESCENDED to 10.83–10.96@36–37k (below the
11.1–11.58 plateau band) — 40k ~16:1x–16:3xZ, reads
via the pre-banked `statedrop_results.py`; SnapFlow @10k probe on GPU
1 ≈ 0.3 GPU-h; **teacher@40k ctrl eval on GPU 1 13:02–13:47Z ≈ 0.75
GPU-h COMPLETE** — 7.1041/2.0720 INSIDE the Amendment 1 band;
GPUs 1–3 otherwise idle by design — reserved for the arch-batch
launches at the arm-C boundary per the posted pre-reg + Amendments,
arm A img280 first).
Explore/exploit: aux-off arm B + noise-floor replicates ≈
instrument/attribution (exploit-side); explore hours proper started
with the noise-draw chain (explore-side, ~9 h queued — pacing check
19:52Z says the draws-10 runs are ~5 h each, so the chain is
longer/richer than planned; still 94–99% util). Literature slice:
on cadence — ~20 min at 22:2xZ (VLM-redundancy + Energy Policy →
Amendment 2) after the ~25 min trunk-survey slice ~19:35–20:00Z;
skipped this 23:12Z session (bounded launch-prep item, slice <1 h
old); then deferred 7 consecutive sessions (00:14–02:4xZ — each had
a ladder-superior item with a launch-path deadline) and **taken
02:4x–02:5xZ (~15 min): ReViP state-dominant-bias mechanism +
state-reliance probe + state-dropout lever banked into #11/#9** —
back on cadence. CPU-side: seven consecutive all-CPU sessions while both GPU
chains ran (trunk survey, flow-vs-AR paired analysis, idea #2a
bucketing, ideas #18.1 hardening, ideas #18.2 reseed-behind-flag,
chunked backward + oracles, E4B checklist prep 23:12Z — ckpt staged
+ CPU parity PASS on the box without touching a GPU) — the
no-idle-pauses rule in action. The
#2a sim result is the rule paying off concretely: a CPU measurement
REPLACED a planned GPU screen (predicted effect sub-threshold —
charter §3). #18.2 keeps the pattern: the instrument break is fully
implemented + pre-registered on CPU; the flip costs one token + one
eval at a boundary we already visit. Sixth consecutive all-CPU
session (#18.8 leakage identity assert ~21:05–21:12Z) continues it.
Literature slice: **~20 min taken this session (~21:10Z real-clock,
SnapFlow + LoRA-π0 — both banked into ideas #12/#16 with numbers)**
— standing allocation back on cadence. Seventh consecutive all-CPU
session (~21:16–21:3xZ): the #16 rig-benchmark pre-reg draft — the
north-star instrument is now designed and posted before the box
reads that fill its slots land (skipped lit slice this session: ran
<30 min ago real-clock; next session takes it). Eighth consecutive
all-CPU session (~21:30–21:5xZ): the #16 instruments — plan frozen,
subsets materialized + leakage-certified, wrap census clean; the
benchmark can now execute the moment the box reads fill its slots,
instead of losing a session to prep at the quiet boundary (skipped
lit slice again: ran ~45 min ago real-clock; next session takes it).
Ninth consecutive all-CPU session (~21:51–22:2xZ): the draws-fairness
instrument — the owner's live 21:49Z challenge went from
in-channel pre-declaration to execution-ready (dump path + frozen
probe + validated reads) before the data it needs finishes
computing; the probe itself costs ~30 GPU-min instead of a ~5 h
full-panel repeat (skipped lit slice: owner-steered item took the
session; the slice is now two sessions overdue — next session MUST
take it). Tenth consecutive all-CPU session (~22:2x–23:0xZ): the
owner-picked E4B pre-reg posted before the box that will run it is
even free, **and the overdue lit slice TAKEN (~20 min: 2606.31382
backbone-redundancy prior banked in #17; Energy Policy 2510.12483 →
the energy-score read pre-declared as Amendment 2 before its data
exists)** — allocation back on cadence. Eleventh consecutive all-CPU
session (22:43–23:1xZ real-clock): chunked backward landed
unconditionally BEFORE the smoke that decides whether it's needed —
the E4B launch path now has no CPU work left on its critical path;
the pre-reg's chunk-mean sketch was corrected by amendment before
any E4B data exists (skipped lit slice: taken last session
real-clock ~22:30Z; next session eligible). Twelfth consecutive
all-CPU session (23:12–23:4xZ real-clock): the stage-2 sign pre-reg
posted (queue's named next item) with feasibility recon done
pre-post, + the draws run-2 headline banked the moment it landed
(mean-of-10 flow 5.365 beats the AR anchor 5.8026) (skipped lit
slice: taken ~1 h ago real-clock; next session eligible).
Thirteenth consecutive all-CPU session (23:37–00:0xZ real-clock):
stage-2 sign probe executed start-to-finish — instrument written,
population + oracle + escalation all inside one GPU-busy window; the
expensive flow decode is cached so the proposed stage-2b amendment
re-runs in minutes (skipped lit slice: taken ~1.5 h ago real-clock;
next session eligible). Fourteenth consecutive all-CPU session
(00:03–00:1xZ real-clock): E4B checklist item 6 — the rsync-back
loop extension whose rotation rule is what keeps the E4B run from
filling the local disk at ~mid-run, done and deployed before the run
that needs it can even launch (skipped lit slice: taken ~1.5 h ago
real-clock and this was a bounded launch-prep item; next session
eligible). Fifteenth consecutive all-CPU session (00:14–00:3xZ
real-clock): **the lit slice WAS the work item** — a targeted
deep-read (SnapFlow recipe extraction + both flagged pointer reads)
converted directly into the #12 SnapFlow distill pre-reg, refilling
the local-GPU queue before its ~09–10Z boundary; allocation on
cadence. Sixteenth consecutive all-CPU session (00:26–00:5xZ
real-clock): the entire SnapFlow impl checklist (5 items) closed in
one GPU-busy window, with validation gate (a) executed on the real
checkpoint and the recipe diff-verified through the real parser —
the run needs only a quiet GPU and the σ_draw amendment (skipped
lit slice: taken last session as the work item itself; next session
eligible). Seventeenth consecutive all-CPU session (00:57–01:1xZ
real-clock): resume hardening (#18.4) — the enforcement landed in
the ~2 h gap before the E4B 100k launch is the first run long
enough to plausibly need a mid-run resume (skipped lit slice: taken
two sessions ago as the work item; next session eligible).
Eighteenth consecutive all-CPU session (01:19–01:4xZ real-clock):
the box-batch results instrument built + four-way oracled in the
~2 h window before its own input data exists, while babysitting
three of the four 40k boundaries live (A-s0 complete + eval
scoring, s1/s2 through their saves) — the ~03–04Z session runs one
command instead of deriving the reads under time pressure (skipped
lit slice: taken three sessions ago as the work item; next session
eligible). Nineteenth consecutive all-CPU session (01:39–02:1xZ
real-clock): the #18.7 duplicate census — the "before trusting fine
holdout deltas" gate — executed start-to-finish in the window
BEFORE the box results read those deltas: 52,507 episodes
fingerprinted, split breach quantified (12.2% of core panel
frames), clean-core anchors banked, all on nice-19 CPU beside five
live eval chains (skipped lit slice: four sessions since the 00:14Z
targeted deep-read — take it next session or state why not).
Twentieth consecutive all-CPU session (02:11–02:4xZ real-clock):
the panel-v2 amendment — the census's follow-on queue item closed
in the window between B's read and the controls' reads, so the
owner can steer the re-definition before the ~04Z boundary where
the noise-key flip (and one bundled re-bank instead of three)
becomes possible (skipped lit slice AGAIN — five sessions since
00:14Z; reason: panel-v2 was the ladder's top unblocked item and
had a real deadline at the ~04Z anchor boundary. The slice is now
firmly overdue: the first session after the box results post MUST
take it as its work item or a named part of one).
Twenty-first consecutive all-CPU session (02:24–02:4xZ real-clock):
the #18.3 Q3 tripwire noise fix — the last deep-dive integrity item
standing on the SnapFlow launch path — landed with a pre-edit banked
bit-exactness oracle in the window before the ~04Z control reads
(lit slice skipped a sixth time; the pure-babysit stretch before
~04Z or the first post-results session takes it — that commitment
stands). Twenty-second consecutive all-CPU session (02:49–03:1xZ
real-clock): the state-reliance probe — last session's lit-slice
mechanism converted into a landed instrument + frozen subset + posted
pre-reg within one session, designed so the intact side pools from
banked npzs and the whole probe costs 1.7 GPU-h in any quiet window
(lit slice: taken last session, ~25 min ago real-clock — on
cadence). Twenty-third consecutive all-CPU session (05:42–06:0xZ
real-clock): the σ_draw finalization amendment — the last CPU-side
blocker on the SnapFlow launch closed in the window while probe arms
3–4 scored, turning five already-banked pooled numbers into both
pre-registered decision bands (no GPU spent; the fairness probe's
direct measurement is the pre-declared cross-check). Lit slice
skipped this session: ~35 min bounded window fully consumed by the
ladder's top item (post-processing a finished run); last slice
02:4x–02:5xZ — next session with slack takes it per the standing
allocation.
Session 06:03–06:3xZ: the state-probe read itself — the 02:4xZ lit
slice's mechanism went pre-reg → instrument → 4 masked runs →
SUPPORTED verdict in ~3.5 h wall-clock end to end (explore-side,
~1.4 GPU-h); the freed GPU went straight to the fairness probe
(instrument-side) per the mantra. Lit slice skipped again — bounded
session, ladder top item; the slice debt stands at the standing
~20–30 min for the next session with slack.
Session 07:20–07:5xZ: the fairness reads — the owner's 21:49Z
challenge went pre-declaration → instrument → probe → verdict in
~10 h wall-clock with every read frozen before its data existed
(instrument/attribution-side, ~1.2 GPU-h incl. the crashed run);
the freed GPU went straight to the #18.2 flip re-bank per the
mantra, gate-asserted against the just-measured σ_draw. Lit slice
skipped — bounded session fully consumed by the ladder's top item
(post-processing a finished run + the chained launch); the ~20–30
min slice debt carries to the next session with slack.
Session 07:51–08:4xZ: the queue-refill work session — #9 state-dropout
went instrument → oracles → pre-reg → LAUNCH in one session (arm C is
**explore-side, ~7.5 GPU-h queued**: real mechanism story, modal
outcome "within band", tail = vision-reliant policy); the re-bank
boundary was taken in-session (ADOPT, anchor 6.5997) and the freed
GPU went straight to SnapFlow (explore-side, ~12–20 h) per the
mantra — both GPUs left busy on explore-class arms. Lit slice: ~10
min taken in the eval-wait window (ThinkProprio + Cloak → #9/#11) —
the standing debt partially serviced; balance carries. Pre-launch
catch worth the surprise log: the SnapFlow launcher's teacher-verbatim
copy had silently inherited a READ-ONLY mainline wandb write target —
the class fix (verify-script pins wandb_project as a named delta) is
in `d9dd385`.
Session 08:5x–09:1xZ: all-CPU while both GPUs trained — the arm-C
results instrument banked before its data (the box-batch
oracle-before-data pattern, third consecutive application:
box-batch → state-probe → state-dropout), so the ~12:4xZ boundary
read is frozen code, not judgment at read time. Lit slice skipped —
bounded session, instrument was the declared queue head; the ~20–30
min standing slice carries to the next session with slack.
Session 09:13–09:4xZ: all-CPU again — the SnapFlow ENDPOINT results
instrument (fourth oracle-before-data application), and the pattern
paid immediately: banking the reads exposed that the live launcher's
chained evals dump no npz, so the pre-reg's per-step horizon read
had no data source — the addendum npz eval is now staged instead of
being improvised at the 13:2xZ boundary. Lit slice skipped — bounded
session, instrument on the critical path (endpoint ~4 h out at
pick time); slice debt now TWO sessions deep — the 10:2xZ probe
babysit window or the first post-endpoint session MUST take it.
Session 09:4x–10:5xZ: the ladder item was #18.5 (rig-rollout safety
gate — CPU, landed + 274 green while both GPUs trained), and the
probe-boundary duty was taken in-session: step_010000 pushed to box
GPU 1 as an expert-only 1.8G rsync (backbone sha256-matched on-box —
the 9G never moved), probe read banked 20 min after the save. **Lit
slice TAKEN (~15 min) — the two-session debt is CLEARED**: the
one-step fallback menu (OFP / MeanFlow-VLA / Let-It-Be-Simple)
banked into #12 ahead of the endpoint read it may steer. Explore
hours: the probe's 0.3 GPU-h is explore-side (SnapFlow chain).
Session 15:13–15:3xZ: the ladder item was post-processing (rung 2) —
the SnapFlow results post filled from the frozen JSON and PUBLISHED
(Space + Discord + owner adoption ask), closing the #12 arc public;
all-CPU (local GPU idle-by-design since the npz addendum banked).
Arm C babysat mid-session with a Discord poll at the checkpoint per
the class fix. Lit slice skipped — bounded publish item, the 13:12Z
session's ~15 min slice is <3 h old; balance carries.
Session 15:43–16:0xZ: the ladder pick was integrity debt (#18.2
default flip, rung 4, ~15 min) — then owner steering (rung 1)
arrived mid-session via the babysit-checkpoint Discord poll and took
the rest: eval-reports hosting + linking, delivered and verified
live in ~35 min. All-CPU (arm C babysat ×2 with polls). Lit slice
skipped — owner-steered session; the 13:12Z slice balance carries.
Session 16:04–16:4xZ: the ladder pick was rung 3 (launching the next
pre-registered run — the arch-batch boundary sequence). GPU-side:
the F1 smokes spent ~0.5 GPU-h ×3 on GPUs 1–3 that were otherwise
idle until the boundary (explore-side: the arch batch bills to the
≥20% budget), overlapped with arm C's chained eval on GPU 0 —
no co-location, and the boundary launch latency dropped from ~1 h
(sync+verify+smoke serial) to minutes (pull+pytest only). Lit slice
TAKEN (~15 min, IVRA → #15) inside the smoke-warmup window.
Session 18:15–18:4xZ: the ladder pick was rung 1 (owner steering —
Molmo2 WP3, confirmed 18:12Z as tonight's critical path); all-CPU
(local GPU idle by design, box GPU 0 on arm C's chained masked
eval). Babysit checkpoint taken mid-session WITH its Discord poll
(class fix holding): caught the owner's 18:18Z probe ask and the
18:34Z multi-image question, both answered in-window; the panel-eval
completion was verified at the same checkpoint (masked eval alive in
scan-warmup, not a stall — 0% GPU was the warmup, checked before
assuming). Lit slice skipped — owner-steered critical-path session
(the 16:04Z slice is <3 h old; balance carries). Explore hours: 0
GPU-h this session; WP3 is exploit-side critical path.
Session 18:41–19:0xZ: the ladder pick was rung 1/2 continuation
(owner-confirmed tonight critical path — WP4 assembly slice + the
18:18Z untrained-gen probe ask). GPU-side: the probe spent ~0.1
GPU-h on the otherwise-idle local GPU (inference burst, the plan's
"parity bursts" allowance — no pre-reg needed, no training). Masked
eval babysat ×2 with Discord polls at boot/checkpoint/close. Lit
slice skipped — critical-path session (the 16:04Z slice balance
carries; tonight's chain outranks). Explore hours: ~0.1 GPU-h,
exploit-side (Molmo2 port is the owner-promoted critical path).
