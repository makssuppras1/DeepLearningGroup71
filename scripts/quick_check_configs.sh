#!/bin/bash
# Ultra-fast config check - shows only key differences
# Usage: ./scripts/quick_check_configs.sh [num_agents]

NUM_AGENTS=${1:-10}
cd "$BLACKHOLE/DeepLearningGroup71" 2>/dev/null || cd "$(dirname "$0")/.."

echo "Quick Config Check (first $NUM_AGENTS agents):"
echo "=============================================="

for i in $(seq 1 $NUM_AGENTS); do
    LOG_FILE="sweep_agent_fashion_mnist_${i}.log"
    if [ -f "$LOG_FILE" ]; then
        CONFIG=$(tail -100 "$LOG_FILE" 2>/dev/null | grep -A 15 "CONFIG FROM WANDB" | grep -E "(optimizer|hidden_layers|learning_rate|batch_size|activation):" | tr '\n' ' | ')
        if [ -n "$CONFIG" ]; then
            printf "Agent %2d: %s\n" "$i" "$CONFIG"
        else
            printf "Agent %2d: [No config found yet]\n" "$i"
        fi
    else
        printf "Agent %2d: [Log not found]\n" "$i"
    fi
done

echo ""
echo "If all agents show the same values, they're running the same config!"
echo "Check WandB dashboard to see if runs have different hyperparameters."

