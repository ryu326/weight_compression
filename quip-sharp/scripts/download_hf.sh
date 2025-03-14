REPO_IDS=(
    # "relaxml/Hessians-Llama-2-7b-6144"
    "relaxml/Hessians-Llama-2-13b-6144"
)

for REPO_ID in "${REPO_IDS[@]}"; do
    python download_hf.py --folder_path ../hess/"$REPO_ID" --repo_id "$REPO_ID" --read_token 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
done