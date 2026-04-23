#!/usr/bin/env bash
set -euo pipefail

uv run pytest -v ./tests --junitxml=test_results.xml || true
echo "Done running tests"

# Set the name of the output zip file
output_file="cs336-spring2025-assignment-5-submission.zip"
rm -f "$output_file"

# Zip only git-tracked files, excluding latex/ and leaderboard.csv
git ls-files \
    | grep -v '^latex/' \
    | grep -v '^cs336_alignment/output/leaderboard.csv' \
    | grep -v '^--annotators_config' \
    | zip "$output_file" -@

echo "All files have been compressed into $output_file"
