#!/usr/bin/env bash
# Continuous rsync-back from the 4xH100 box (temporary-box discipline,
# pre-reg 2026-08-05-prereg-box-batch-4xh100.md): logs + eval reports
# always; latest two saves per run. Cadence 20 min.
#
# E4B extension (checklist item 6, posts/2026-08-05-prereg-e4b-screen.md):
# - fontaine_arb_rcond_e4b_100k_ddp4 added to RUNS (its train/eval logs
#   already match the log globs).
# - Partial-copy guard: a step dir gets a .synced_complete marker only
#   when a follow-up --dry-run transfers nothing, so the local E4/E5
#   panel evals can refuse a mid-save copy.
# - Panel-step repair: E4B steps {025000,050000,100000} are re-synced
#   until marked complete even after they leave the latest-2 window
#   (the box keeps all saves; only the local copy rotates).
# - Local rotation, E4B ONLY: keep latest two + the panel steps. E4B
#   saves are ~35-40 GB x 40; unpruned local accumulation (~1.5 T)
#   would exhaust local free space. The four 40k-run copies are never
#   pruned.
BOX=ubuntu@192.222.55.210
# 2026-08-06 23:4xZ: e4b dropped from RUNS — its box dir is gone
# (retired screen; every pass emitted a remote glob error). molmo2 AR
# 40k added (live run, saves every 2,500 from ~00:4xZ) with an
# E4B-style local rotation: latest two + the 40k endpoint, pruned
# otherwise (consolidated zero1 saves are large; the endpoint also
# uploads to fontaine-checkpoints at the boundary per standing rule).
RUNS="fontaine_arb_rcond_40k_1xh100 fontaine_arb_rcond_auxoff_40k_1xh100 fontaine_arb_rcond_40k_1xh100_s1 fontaine_arb_rcond_40k_1xh100_s2 fontaine_molmo2_ar_40k_ddp4"
E4B=fontaine_arb_rcond_e4b_100k_ddp4
E4B_KEEP="step_025000 step_050000 step_100000"
MOLMO2=fontaine_molmo2_ar_40k_ddp4
MOLMO2_KEEP="step_040000"

sync_step() { # sync_step <run> <box_step_dir>
  local r=$1 d=$2 s
  s=$(basename "$d")
  mkdir -p ~/boxsync/outputs/$r
  rsync -a $BOX:"$d" ~/boxsync/outputs/$r/ 2>/dev/null
  if [ ! -e ~/boxsync/outputs/$r/$s/.synced_complete ] && \
     ! rsync -ai --dry-run $BOX:"$d" ~/boxsync/outputs/$r/ 2>/dev/null | grep -q .; then
    touch ~/boxsync/outputs/$r/$s/.synced_complete
    echo "  $r/$s synced_complete"
  fi
}

while true; do
  date -u +"%FT%TZ rsync pass"
  rsync -a --include='train_fontaine_*.log' --include='eval_fontaine_*.log' --exclude='*' $BOX:'~/' ~/boxsync/logs/ 2>/dev/null
  rsync -a $BOX:'~/flow-matching/reports/eval__fontaine_*' ~/boxsync/reports/ 2>/dev/null
  for r in $RUNS; do
    for d in $(ssh $BOX "ls -d ~/flow-matching/outputs/train/$r/step_* 2>/dev/null | sort | tail -2"); do
      sync_step "$r" "$d"
    done
  done
  # E4B panel-step repair: complete any keep-step the latest-2 window
  # left partial (box keeps every save, so the source is always there).
  for s in $E4B_KEEP; do
    if [ ! -e ~/boxsync/outputs/$E4B/$s/.synced_complete ] && \
       ssh $BOX "test -d ~/flow-matching/outputs/train/$E4B/$s" 2>/dev/null; then
      sync_step "$E4B" "~/flow-matching/outputs/train/$E4B/$s"
    fi
  done
  # E4B local rotation: latest two + panel steps, everything else pruned.
  if [ -d ~/boxsync/outputs/$E4B ]; then
    keep="$(ls -d ~/boxsync/outputs/$E4B/step_* 2>/dev/null | sort | tail -2)"
    for p in $(ls -d ~/boxsync/outputs/$E4B/step_* 2>/dev/null); do
      s=$(basename "$p")
      case " $E4B_KEEP " in *" $s "*) continue ;; esac
      echo "$keep" | grep -qx "$p" || { echo "  prune local $E4B/$s"; rm -rf "$p"; }
    done
  fi
  # molmo2 local rotation: latest two + the endpoint, everything else
  # pruned (same shape as the E4B rotation above).
  if [ -d ~/boxsync/outputs/$MOLMO2 ]; then
    keep="$(ls -d ~/boxsync/outputs/$MOLMO2/step_* 2>/dev/null | sort | tail -2)"
    for p in $(ls -d ~/boxsync/outputs/$MOLMO2/step_* 2>/dev/null); do
      s=$(basename "$p")
      case " $MOLMO2_KEEP " in *" $s "*) continue ;; esac
      echo "$keep" | grep -qx "$p" || { echo "  prune local $MOLMO2/$s"; rm -rf "$p"; }
    done
  fi
  sleep 1200
done
