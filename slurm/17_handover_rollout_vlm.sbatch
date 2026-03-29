#!/bin/bash
#SBATCH --job-name=nlpfp_handover_vlm
#SBATCH --account=torch_pr_637_tandon_priority
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

WORKDIR="${WORKDIR:-$SLURM_SUBMIT_DIR}"
CONDA_ENV_DIR="${CONDA_ENV_DIR:-$WORKDIR/.conda_torch}"
PYTHON_BIN="${PYTHON_BIN:-$CONDA_ENV_DIR/bin/python}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/bj2410}"
XDG_CACHE_HOME="${XDG_CACHE_HOME:-$SCRATCH_ROOT/.cache}"
MPLCONFIGDIR="${MPLCONFIGDIR:-$SCRATCH_ROOT/.cache/matplotlib}"
MESA_SHADER_CACHE_DIR="${MESA_SHADER_CACHE_DIR:-$SCRATCH_ROOT/.cache/mesa_shader_cache}"
PYTHONPATH_EXTRA="${PYTHONPATH_EXTRA:-$WORKDIR/third_party/handover-sim:$WORKDIR/third_party/handover-sim/examples}"

MODEL_NAME="${MODEL_NAME:-/scratch/bj2410/models/Qwen2.5-VL-3B-Instruct}"
ADAPTER_PATH="${ADAPTER_PATH:-artifacts/schema_vlm_qwen25vl3b_direct/best}"
TASK_PROFILE="${TASK_PROFILE:-}"
INSTRUCTION_TEMPLATE="${INSTRUCTION_TEMPLATE:-}"
TARGET_OBJECT_KEY="${TARGET_OBJECT_KEY:-}"
TARGET_OBJECT_INDEX="${TARGET_OBJECT_INDEX:-0}"
INTERVENTION_MODE="${INTERVENTION_MODE:-minimal}"
SETUP_NAME="${SETUP_NAME:-s0}"
SPLIT_NAME="${SPLIT_NAME:-val}"
SCENE_INDEX="${SCENE_INDEX:-0}"
SCENE_INDICES="${SCENE_INDICES:-}"
SCENE_INDICES_STR="${SCENE_INDICES_STR:-}"
ENV_ID="${ENV_ID:-HandoverStateEnv-v1}"
POLICY_TIME_WAIT="${POLICY_TIME_WAIT:-0.0}"
CONTACT_FORCE_THRESH="${CONTACT_FORCE_THRESH:-0.5}"
RELEASE_FORCE_THRESH="${RELEASE_FORCE_THRESH:-0.5}"
MANO_RESTITUTION="${MANO_RESTITUTION:-0.1}"
IMAGE_SIZE="${IMAGE_SIZE:-448}"
MAX_LENGTH="${MAX_LENGTH:-512}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-96}"
VLM_INTERVAL="${VLM_INTERVAL:-25}"
ENABLE_EVENT_TRIGGER="${ENABLE_EVENT_TRIGGER:-1}"
EVENT_DGRIP_DELTA="${EVENT_DGRIP_DELTA:-0.01}"
EVENT_HAND_OBJ_DELTA="${EVENT_HAND_OBJ_DELTA:-0.012}"
MAX_JOINT_STEP="${MAX_JOINT_STEP:-0.03}"
SLOWDOWN_DIST="${SLOWDOWN_DIST:-0.12}"
NEAR_STEP_SCALE="${NEAR_STEP_SCALE:-0.35}"
ASSIST_CLOSE_DIST="${ASSIST_CLOSE_DIST:-0.06}"
ASSIST_CLOSE_RELEASE_DIST="${ASSIST_CLOSE_RELEASE_DIST:-0.085}"
ASSIST_CLOSE_HOLD_FRAMES="${ASSIST_CLOSE_HOLD_FRAMES:-90}"
GRASP_LOCK_ENTER_DIST="${GRASP_LOCK_ENTER_DIST:-0.09}"
GRASP_LOCK_EXIT_DIST="${GRASP_LOCK_EXIT_DIST:-0.13}"
GRASP_LOCK_HOLD_FRAMES="${GRASP_LOCK_HOLD_FRAMES:-220}"
FORCE_GRASP_DIST="${FORCE_GRASP_DIST:-0.06}"
FORCE_GRASP_HOLD_FRAMES="${FORCE_GRASP_HOLD_FRAMES:-35}"
FORCE_GRASP_LATCH_DIST="${FORCE_GRASP_LATCH_DIST:-0.09}"
CLOSE_STAGE_ENTER_DIST="${CLOSE_STAGE_ENTER_DIST:-0.09}"
CLOSE_STAGE_HOLD_FRAMES="${CLOSE_STAGE_HOLD_FRAMES:-8}"
CLOSE_STAGE_MAX_FRAMES="${CLOSE_STAGE_MAX_FRAMES:-80}"
DISABLE_GEOM_GATE="${DISABLE_GEOM_GATE:-0}"
GEOM_CONTACT_MAX_DIST="${GEOM_CONTACT_MAX_DIST:-0.05}"
GEOM_SECURE_MAX_DIST="${GEOM_SECURE_MAX_DIST:-0.04}"
DISABLE_HUMAN_READY_GATE="${DISABLE_HUMAN_READY_GATE:-0}"
HUMAN_READY_MAX_DIST="${HUMAN_READY_MAX_DIST:-0.16}"
DISABLE_HUMAN_STABLE_GATE="${DISABLE_HUMAN_STABLE_GATE:-0}"
HUMAN_STABLE_MAX_SPEED="${HUMAN_STABLE_MAX_SPEED:-0.10}"
HUMAN_STABLE_MIN_FRAMES="${HUMAN_STABLE_MIN_FRAMES:-16}"
APPROACH_KEEPOUT_DIST="${APPROACH_KEEPOUT_DIST:-0.14}"
HANDOVER_HAND_OBJ_NEAR_DIST="${HANDOVER_HAND_OBJ_NEAR_DIST:-0.12}"
NO_APPROACH_MOVING_DIST="${NO_APPROACH_MOVING_DIST:-0.35}"
MOTION_LOCK_RELEASE_STABLE_FRAMES="${MOTION_LOCK_RELEASE_STABLE_FRAMES:-45}"
MOTION_LOCK_OBJ_SPEED_MAX="${MOTION_LOCK_OBJ_SPEED_MAX:-0.03}"
SCHEMA_REACQUIRE_RETREAT_STEP="${SCHEMA_REACQUIRE_RETREAT_STEP:-0.015}"
PRE_RELEASE_KEEPOUT_DIST="${PRE_RELEASE_KEEPOUT_DIST:-0.20}"
RELEASE_FALLBACK_FRAME="${RELEASE_FALLBACK_FRAME:-3200}"
WAIT_HOVER_Z_OFFSET="${WAIT_HOVER_Z_OFFSET:-0.18}"
WAIT_HOVER_XY_BLEND="${WAIT_HOVER_XY_BLEND:-0.70}"
WAIT_HOVER_MAX_JOINT_STEP="${WAIT_HOVER_MAX_JOINT_STEP:-0.02}"
WAIT_UPDATE_INTERVAL="${WAIT_UPDATE_INTERVAL:-12}"
WAIT_TARGET_DEADBAND="${WAIT_TARGET_DEADBAND:-0.03}"
HUMAN_READY_MAX_WAIT_FRAME="${HUMAN_READY_MAX_WAIT_FRAME:-500}"
MAX_VLM_CALLS="${MAX_VLM_CALLS:-24}"
MIN_VLM_GAP="${MIN_VLM_GAP:-20}"
NEAR_THRESH="${NEAR_THRESH:-0.18}"
SAFETY_HAND_KEEPOUT_DIST="${SAFETY_HAND_KEEPOUT_DIST:-0.12}"
NEAR_CONTACT_COMMIT_DIST="${NEAR_CONTACT_COMMIT_DIST:-0.052}"
NEAR_CONTACT_COMMIT_HOLD_FRAMES="${NEAR_CONTACT_COMMIT_HOLD_FRAMES:-22}"
COMMIT_MIN_FRAME="${COMMIT_MIN_FRAME:-120}"
COMMIT_VLM_VOTES="${COMMIT_VLM_VOTES:-2}"
FORCE_COMMIT_FRAME="${FORCE_COMMIT_FRAME:-900}"
HARD_FORCE_COMMIT_FRAME="${HARD_FORCE_COMMIT_FRAME:-2600}"
FORCE_ABORT_FRAME="${FORCE_ABORT_FRAME:-12000}"
COLD_START_MODE="${COLD_START_MODE:-near}"
ENABLE_NON_TARGET_KEEPOUT="${ENABLE_NON_TARGET_KEEPOUT:-0}"
NON_TARGET_KEEPOUT_DIST="${NON_TARGET_KEEPOUT_DIST:-0.09}"
USE_SCHEMA_CHECKER="${USE_SCHEMA_CHECKER:-1}"
USE_CONSISTENCY_RERANK="${USE_CONSISTENCY_RERANK:-1}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"
BF16="${BF16:-1}"
FP16="${FP16:-0}"
RES_DIR="${RES_DIR:-artifacts/table_b_rollout/vlm_full_smoke}"
OUT_PATH="${OUT_PATH:-$RES_DIR/summary.json}"
SAVE_VIDEO="${SAVE_VIDEO:-0}"
VIDEO_FPS="${VIDEO_FPS:-20}"
VIDEO_EVERY_N="${VIDEO_EVERY_N:-1}"
VIDEO_MAX_FRAMES="${VIDEO_MAX_FRAMES:-0}"

cd "$WORKDIR"
export XDG_CACHE_HOME MPLCONFIGDIR MESA_SHADER_CACHE_DIR PYTHONPATH="$PYTHONPATH_EXTRA:${PYTHONPATH:-}"
mkdir -p "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$MESA_SHADER_CACHE_DIR" "$RES_DIR"

CMD=(
  "$PYTHON_BIN" scripts/run_handover_vlm_rollout.py
  --model-name "$MODEL_NAME"
  --adapter-path "$ADAPTER_PATH"
  --target-object-index "$TARGET_OBJECT_INDEX"
  --intervention-mode "$INTERVENTION_MODE"
  --setup "$SETUP_NAME"
  --split "$SPLIT_NAME"
  --scene-index "$SCENE_INDEX"
  --env-id "$ENV_ID"
  --policy-time-wait "$POLICY_TIME_WAIT"
  --contact-force-thresh "$CONTACT_FORCE_THRESH"
  --release-force-thresh "$RELEASE_FORCE_THRESH"
  --mano-restitution "$MANO_RESTITUTION"
  --res-dir "$RES_DIR"
  --out-path "$OUT_PATH"
  --video-fps "$VIDEO_FPS"
  --video-every-n "$VIDEO_EVERY_N"
  --video-max-frames "$VIDEO_MAX_FRAMES"
  --image-size "$IMAGE_SIZE"
  --max-length "$MAX_LENGTH"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --vlm-interval "$VLM_INTERVAL"
  --event-dgrip-delta "$EVENT_DGRIP_DELTA"
  --event-hand-obj-delta "$EVENT_HAND_OBJ_DELTA"
  --max-joint-step "$MAX_JOINT_STEP"
  --slowdown-dist "$SLOWDOWN_DIST"
  --near-step-scale "$NEAR_STEP_SCALE"
  --assist-close-dist "$ASSIST_CLOSE_DIST"
  --assist-close-release-dist "$ASSIST_CLOSE_RELEASE_DIST"
  --assist-close-hold-frames "$ASSIST_CLOSE_HOLD_FRAMES"
  --grasp-lock-enter-dist "$GRASP_LOCK_ENTER_DIST"
  --grasp-lock-exit-dist "$GRASP_LOCK_EXIT_DIST"
  --grasp-lock-hold-frames "$GRASP_LOCK_HOLD_FRAMES"
  --force-grasp-dist "$FORCE_GRASP_DIST"
  --force-grasp-hold-frames "$FORCE_GRASP_HOLD_FRAMES"
  --force-grasp-latch-dist "$FORCE_GRASP_LATCH_DIST"
  --close-stage-enter-dist "$CLOSE_STAGE_ENTER_DIST"
  --close-stage-hold-frames "$CLOSE_STAGE_HOLD_FRAMES"
  --close-stage-max-frames "$CLOSE_STAGE_MAX_FRAMES"
  --human-ready-max-dist "$HUMAN_READY_MAX_DIST"
  --human-stable-max-speed "$HUMAN_STABLE_MAX_SPEED"
  --human-stable-min-frames "$HUMAN_STABLE_MIN_FRAMES"
  --approach-keepout-dist "$APPROACH_KEEPOUT_DIST"
  --handover-hand-obj-near-dist "$HANDOVER_HAND_OBJ_NEAR_DIST"
  --no-approach-moving-dist "$NO_APPROACH_MOVING_DIST"
  --motion-lock-release-stable-frames "$MOTION_LOCK_RELEASE_STABLE_FRAMES"
  --motion-lock-obj-speed-max "$MOTION_LOCK_OBJ_SPEED_MAX"
  --schema-reacquire-retreat-step "$SCHEMA_REACQUIRE_RETREAT_STEP"
  --pre-release-keepout-dist "$PRE_RELEASE_KEEPOUT_DIST"
  --release-fallback-frame "$RELEASE_FALLBACK_FRAME"
  --wait-hover-z-offset "$WAIT_HOVER_Z_OFFSET"
  --wait-hover-xy-blend "$WAIT_HOVER_XY_BLEND"
  --wait-hover-max-joint-step "$WAIT_HOVER_MAX_JOINT_STEP"
  --wait-update-interval "$WAIT_UPDATE_INTERVAL"
  --wait-target-deadband "$WAIT_TARGET_DEADBAND"
  --human-ready-max-wait-frame "$HUMAN_READY_MAX_WAIT_FRAME"
  --geom-contact-max-dist "$GEOM_CONTACT_MAX_DIST"
  --geom-secure-max-dist "$GEOM_SECURE_MAX_DIST"
  --max-vlm-calls "$MAX_VLM_CALLS"
  --min-vlm-gap "$MIN_VLM_GAP"
  --near-thresh "$NEAR_THRESH"
  --safety-hand-keepout-dist "$SAFETY_HAND_KEEPOUT_DIST"
  --near-contact-commit-dist "$NEAR_CONTACT_COMMIT_DIST"
  --near-contact-commit-hold-frames "$NEAR_CONTACT_COMMIT_HOLD_FRAMES"
  --commit-min-frame "$COMMIT_MIN_FRAME"
  --commit-vlm-votes "$COMMIT_VLM_VOTES"
  --force-commit-frame "$FORCE_COMMIT_FRAME"
  --hard-force-commit-frame "$HARD_FORCE_COMMIT_FRAME"
  --force-abort-frame "$FORCE_ABORT_FRAME"
  --non-target-keepout-dist "$NON_TARGET_KEEPOUT_DIST"
  --cold-start-mode "$COLD_START_MODE"
)

if [[ "$USE_SCHEMA_CHECKER" == "1" ]]; then
  CMD+=(--use-schema-checker)
fi
if [[ "$USE_CONSISTENCY_RERANK" == "1" ]]; then
  CMD+=(--use-consistency-rerank)
fi
if [[ "$ENABLE_EVENT_TRIGGER" == "1" ]]; then
  CMD+=(--enable-event-trigger)
fi
if [[ "$ENABLE_NON_TARGET_KEEPOUT" == "1" ]]; then
  CMD+=(--enable-non-target-keepout)
fi
if [[ "$DISABLE_GEOM_GATE" == "1" ]]; then
  CMD+=(--disable-geom-gate)
fi
if [[ "$DISABLE_HUMAN_READY_GATE" == "1" ]]; then
  CMD+=(--disable-human-ready-gate)
fi
if [[ "$DISABLE_HUMAN_STABLE_GATE" == "1" ]]; then
  CMD+=(--disable-human-stable-gate)
fi
if [[ "$TRUST_REMOTE_CODE" == "1" ]]; then
  CMD+=(--trust-remote-code)
fi
if [[ "$BF16" == "1" ]]; then
  CMD+=(--bf16)
fi
if [[ "$FP16" == "1" ]]; then
  CMD+=(--fp16)
fi
if [[ "$SAVE_VIDEO" == "1" ]]; then
  CMD+=(--save-video)
fi
if [[ -n "$TASK_PROFILE" ]]; then
  CMD+=(--task-profile "$TASK_PROFILE")
fi
if [[ -n "$INSTRUCTION_TEMPLATE" ]]; then
  CMD+=(--instruction-template "$INSTRUCTION_TEMPLATE")
fi
if [[ -n "$TARGET_OBJECT_KEY" ]]; then
  CMD+=(--target-object-key "$TARGET_OBJECT_KEY")
fi
if [[ -n "$SCENE_INDICES_STR" ]]; then
  SCENE_INDICES_CSV="${SCENE_INDICES_STR//:/,}"
  CMD+=(--scene-indices "$SCENE_INDICES_CSV")
elif [[ -n "$SCENE_INDICES" ]]; then
  CMD+=(--scene-indices "$SCENE_INDICES")
fi

echo "Running command:"
printf '%q ' "${CMD[@]}"
echo
"${CMD[@]}"
