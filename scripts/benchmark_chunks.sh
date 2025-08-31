#!/bin/bash

URL="http://localhost:8000/generate_images"

PROMPTS='["a cute baby dragon sitting on a pile of gold",
"a cyberpunk city skyline at night with neon lights",
"a magical forest with glowing mushrooms and fairies",
"an astronaut riding a horse on Mars",
"a realistic portrait of a medieval knight in armor",
"a futuristic flying car above a sci-fi city",
"a panda surfing a giant wave, digital art",
"a castle floating in the clouds, fantasy style",
"a robot painting a picture in Van Gogh style",
"an ancient library filled with floating books and candles",
"a cute baby dragon sitting on a pile of gold",
"a cyberpunk city skyline at night with neon lights",
"a magical forest with glowing mushrooms and fairies",
"an astronaut riding a horse on Mars",
"a realistic portrait of a medieval knight in armor",
"a futuristic flying car above a sci-fi city",
"a panda surfing a giant wave, digital art",
"a castle floating in the clouds, fantasy style",
"a robot painting a picture in Van Gogh style",
"an ancient library filled with floating books and candles"]'

COUNT=20

for CHUNK in 1 2 5
do
  echo "=== chunk_size=$CHUNK ==="
  START=$(date +%s)
  curl -s -X POST "$URL" \
    -H "Content-Type: application/json" \
    -d "{\"prompts\": $PROMPTS, \"chunk_size\": $CHUNK}" \
    -o out_${CHUNK}.zip
  END=$(date +%s)
  ELAPSED=$((END - START))
  if [ $ELAPSED -gt 0 ]; then
    THROUGHPUT=$(echo "scale=2; $COUNT / $ELAPSED" | bc)
  else
    THROUGHPUT="inf"
  fi
  echo "Count: $COUNT"
  echo "Time: ${ELAPSED}s"
  echo "Throughput: ${THROUGHPUT} img/sec"
  echo
done
