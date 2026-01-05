#!/bin/bash

# API credentials
API_KEY="2e6f7c530b3935dd3c922b306d24ff56"
SECRET_KEY="ebf68ed6602e04793701f9f98fb63cb2"

# Generate nonce (32 hex chars = 16 bytes)
NONCE=$(openssl rand -hex 16)

# Generate timestamp (milliseconds since epoch)
TIMESTAMP=$(date +%s%3N)

# Query parameters (sorted alphabetically)
QUERY_PARAMS="symbolBTCUSDT"

# Body is empty for GET request
BODY=""

# Create digest input
DIGEST_INPUT="${NONCE}${TIMESTAMP}${API_KEY}${QUERY_PARAMS}${BODY}"

# First SHA256 hash
FIRST_HASH=$(echo -n "$DIGEST_INPUT" | openssl sha256 -hex | cut -d' ' -f2)

# Second SHA256 hash with secret key
SIGN=$(echo -n "${FIRST_HASH}${SECRET_KEY}" | openssl sha256 -hex | cut -d' ' -f2)

# Make the API call
curl -X 'GET' \
  --location 'https://fapi.bitunix.com/api/v1/futures/position/get_history_positions?symbol=BTCUSDT' \
  -H "api-key: ${API_KEY}" \
  -H "sign: ${SIGN}" \
  -H "nonce: ${NONCE}" \
  -H "timestamp: ${TIMESTAMP}" \
  -H "language: en-US" \
  -H "Content-Type: application/json"
