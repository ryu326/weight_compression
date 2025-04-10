# nohup bash scripts/run0.sh > ./logs/run0.log 2>&1 &
nohup bash scripts/run1.sh > ./logs/run1.log 2>&1 &
nohup bash scripts/run2.sh > ./logs/run2.log 2>&1 &
# nohup bash scripts/run3.sh > ./logs/run3.log 2>&1 &

wait