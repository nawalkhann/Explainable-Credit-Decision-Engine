#!/usr/bin/env bash
# Downloads the public datasets used by train.py into this folder.
set -euo pipefail
cd "$(dirname "$0")"
curl -sSL -o lending_club_loans.csv https://raw.githubusercontent.com/h2oai/app-consumer-loan/master/data/loan.csv
curl -sSL -o german_credit.csv https://raw.githubusercontent.com/selva86/datasets/master/GermanCredit.csv
echo "Downloaded: $(ls *.csv | tr '\n' ' ')"
