#!/bin/bash

URL="http://localhost:8000/generate"

PROMPTS=(
  "a cute baby dragon sitting on a pile of gold"
  "a cyberpunk city skyline at night with neon lights"
  "a magical forest with glowing mushrooms and fairies"
  "an astronaut riding a horse on Mars"
  "a realistic portrait of a medieval knight in armor"
  "a futuristic flying car above a sci-fi city"
  "a panda surfing a giant wave, digital art"
  "a castle floating in the clouds, fantasy style"
  "a robot painting a picture in Van Gogh style"
  "an ancient library filled with floating books and candles"
  "a cute baby dragon sitting on a pile of gold"
  "a cyberpunk city skyline at night with neon lights"
  "a magical forest with glowing mushrooms and fairies"
  "an astronaut riding a horse on Mars"
  "a realistic portrait of a medieval knight in armor"
  "a futuristic flying car above a sci-fi city"
  "a panda surfing a giant wave, digital art"
  "a castle floating in the clouds, fantasy style"
  "a robot painting a picture in Van Gogh style"
  "an ancient library filled with floating books and candles"
)

COUNT=${#PROMPTS[@]}
START=$(date +%s)

for i in "${!PROMPTS[@]}"; do
  PROMPT=${PROMPTS[$i]}
  curl -s -G "$URL" \
    --data-urlencode "prompt=$PROMPT" \
    -o out_${i}.png
done

END=$(date +%s)
ELAPSED=$((END - START))
THROUGHPUT=$(echo "scale=2; $COUNT / $ELAPSED" | bc)

echo "=== /generate benchmark ==="
echo "Count: $COUNT"
echo "Time: ${ELAPSED}s"
echo "Throughput: ${THROUGHPUT} img/sec"
