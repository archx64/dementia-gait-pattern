export LD_LIBRARY_PATH="/home/aicenter/miniconda3/envs/openmmlab/lib/python3.8/site-packages/torch/lib:$LD_LIBRARY_PATH"

TOTAL=$(python -c "import yaml; print(yaml.safe_load(open('config/session.yaml'))['total_rounds'])")
for r in $(seq 1 $TOTAL); do
  echo "===== ROUND $r / $TOTAL ====="
  python -c "
import yaml
with open('config/session.yaml') as f:
    cfg = yaml.safe_load(f)
cfg['round'] = $r
with open('config/session.yaml', 'w') as f:
    yaml.dump(cfg, f)
"
  python -um src.pose_estimation_v2 && python -um src.gait_analysis_v2
  if [ $? -ne 0 ]; then
    echo "ERROR: round $r failed, stopping."
    break
  fi
done
