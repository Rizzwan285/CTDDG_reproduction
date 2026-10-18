#!/bin/sh
# Offline test suite for the CTDDG notebooks. CPU only, no data, no GPU, seconds.
# Run after editing any notebook and before any long job.
set -e
cd "$(dirname "$0")"
python3 99_consistency_audit.py
for t in test_00_selftests.py test_02.py test_03.py test_04.py test_05_review_fixes.py; do
    echo
    echo "===== $t ====="
    python3 -I "$t"
done
echo
echo "All suites passed."
