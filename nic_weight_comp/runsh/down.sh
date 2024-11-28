# 파일 ID 배열
file_ids=(
    "https://drive.google.com/file/d/1Quz6_jGJyaG6LMUbT4JuOhhQWxJN26Kh/view?usp=sharing"
    "https://drive.google.com/file/d/1rc4E2Rke1Jd8UnLq73NaXbfAdcBGGPKg/view?usp=sharing"
    "https://drive.google.com/file/d/1UbfQFsrr-Z6SrvZvpX4p1QPta5FCORZ5/view?usp=sharing"
    "https://drive.google.com/file/d/17THA1IiPStSO6jG4h5clwkw0ySzgLZID/view?usp=sharing"
    "https://drive.google.com/file/d/1x2rfIQAv8RsjM3zEByDdOZJtEcPU5XZT/view?usp=sharing"
    "https://drive.google.com/file/d/1zpkW_MCkUWl8nRUlza0L7Fk7dXlXciZd/view?usp=sharing"
)

# 각 파일을 병렬로 다운로드
for file_url in "${file_ids[@]}"; do
    # URL에서 파일 ID 추출
    file_id=$(echo $file_url | sed -E 's|https://drive.google.com/file/d/([^/]+)/.*|\1|')
    gdown "https://drive.google.com/uc?id=$file_id" &
done

# 모든 백그라운드 프로세스가 종료될 때까지 기다림
wait
