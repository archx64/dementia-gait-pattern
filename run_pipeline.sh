source dem-env/bin/activate
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
  bash pega.sh
  if [ $? -ne 0 ]; then
    echo "ERROR: round $r failed, stopping."
    break
  fi
done
