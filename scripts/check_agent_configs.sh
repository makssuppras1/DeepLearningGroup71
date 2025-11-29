#!/bin/bash
# Quick script to check if different agents are getting different configs
# Usage: ./scripts/check_agent_configs.sh [num_agents_to_check]

NUM_AGENTS=${1:-5}

echo "Checking configs from $NUM_AGENTS agents..."
echo "=========================================="
echo ""

for i in $(seq 1 $NUM_AGENTS); do
    LOG_FILE="sweep_agent_fashion_mnist_${i}.log"
    if [ -f "$LOG_FILE" ]; then
        echo "=== Agent $i ==="
        # Extract the key hyperparameters
        tail -100 "$LOG_FILE" 2>/dev/null | grep -A 15 "CONFIG FROM WANDB" | grep -E "(optimizer|hidden_layers|learning_rate|batch_size|activation|l2_lambda|num_epochs|weight_init|dropout_rate):" | head -9
        echo ""
    else
        echo "=== Agent $i ==="
        echo "Log file not found: $LOG_FILE"
        echo ""
    fi
done

echo "=========================================="
echo "Summary: Compare the values above - they should be different!"
echo ""
echo "To see full configs, run:"
echo "  tail -50 sweep_agent_fashion_mnist_1.log | grep -A 20 'CONFIG FROM WANDB'"
echo "  tail -50 sweep_agent_fashion_mnist_2.log | grep -A 20 'CONFIG FROM WANDB'"
echo "  # etc..."

