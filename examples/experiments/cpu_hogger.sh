#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_FILE="$SCRIPT_DIR/cpu_hogger.csv"

start_ts=$(date +%s.%N)

./bin/serverledge-cli invoke -f cpu_hogger -p 'duration:180' -p 'memory:800'

end_ts=$(date +%s.%N)

if [[ ! -s "$OUTPUT_FILE" ]]; then
    printf "start_ts,end_ts\n" >> "$OUTPUT_FILE"
fi

printf "%s,%s\n" "$start_ts" "$end_ts" >> "$OUTPUT_FILE"



echo "Done"
