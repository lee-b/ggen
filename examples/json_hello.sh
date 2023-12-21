#!/bin/sh

PROMPT="Say hello as a single, simple json message, with one message field."

MODEL=~/AI/models/Mistral-7B-Instruct-v0.2-GGUF/mistral-7b-instruct-v0.2.Q6_K.gguf
PROMPT_TEMPLATE=mistral-instruct
SCHEMA=json

ggen "$PROMPT" -t "$PROMPT_TEMPLATE" -m "$MODEL" -s "$SCHEMA"

