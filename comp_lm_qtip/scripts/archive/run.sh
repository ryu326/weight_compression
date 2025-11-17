nohup bash scripts/comp_lm.sh > ./log/comp_lm.out 2>&1 &
nohup bash scripts/comp_lm1.sh > ./log/comp_lm1.out 2>&1 &
nohup bash scripts/comp_lm2.sh > ./log/comp_lm2.out 2>&1 &
nohup bash scripts/comp_lm3.sh > ./log/comp_lm3.out 2>&1 &

nohup bash scripts/naive/comp_lm_jpeg.sh > ./log/comp_lm_jpeg.out 2>&1 &
nohup bash scripts/naive/comp_lm_jpeg1.sh > ./log/comp_lm_jpeg1.out 2>&1 &
nohup bash scripts/naive/comp_lm_jpeg2.sh > ./log/comp_lm_jpeg2.out 2>&1 &
nohup bash scripts/naive/comp_lm_jpeg3.sh > ./log/comp_lm_jpeg3.out 2>&1 &

nohup bash scripts/naive/comp_lm_webp.sh > ./log/comp_lm_webp.out 2>&1 &
nohup bash scripts/naive/comp_lm_webp1.sh > ./log/comp_lm_webp1.out 2>&1 &
nohup bash scripts/naive/comp_lm_webp2.sh > ./log/comp_lm_webp2.out 2>&1 &
nohup bash scripts/naive/comp_lm_webp3.sh > ./log/comp_lm_webp3.out 2>&1 &
