import numpy as np
import matplotlib.pyplot as plt
import yaml
from yaml import Loader

# Load configuration file to get parent_c and parent_k
stream = open("configs/ga0.yaml", 'r')
cnfg = yaml.load(stream, Loader=Loader)

parent_c = cnfg['parent_c'] 
parent_k = cnfg['parent_k'] 
sol_per_pop = cnfg['sol_per_pop']

# Create ranks (from 1 to sol_per_pop)
ranks = np.arange(1, sol_per_pop + 1)

# Compute probabilities using exponential rank selection
probs = parent_c * (1 - np.exp(-ranks / parent_k))

# Normalize probabilities
probs_normalized = probs / np.sum(probs)

# Create the plot
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

# Plot normalized probabilities
ax.plot(ranks, probs_normalized, 'r-', linewidth=2, label='Selection Probability')
ax.scatter(ranks[::max(1, sol_per_pop//20)], probs_normalized[::max(1, sol_per_pop//20)], 
            c='red', s=50, alpha=0.6, zorder=5)
ax.set_xlabel('Rank (1 = Best)', fontsize=12)
ax.set_ylabel('Selection Probability', fontsize=12)
ax.set_title(f'Exponential Rank Selection\n(c={parent_c}, k={parent_k}, Sum = {np.sum(probs_normalized):.4f})', 
              fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)

# Add text box with statistics
stats_text = f'Population Size: {sol_per_pop}\n'
stats_text += f'Best Individual Prob: {probs_normalized[0]:.4f}\n'
stats_text += f'Worst Individual Prob: {probs_normalized[-1]:.6f}\n'
stats_text += f'Top 10% Cumulative Prob: {np.sum(probs_normalized[:max(1, sol_per_pop//10)]):.4f}'

ax.text(0.98, 0.02, stats_text, transform=ax.transAxes,
         fontsize=9, verticalalignment='bottom', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('rank_selection_probability.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Plot saved as 'rank_selection_probability.png'")
plt.show()

# Print additional statistics
print(f"\n{'='*60}")
print(f"EXPONENTIAL RANK SELECTION ANALYSIS")
print(f"{'='*60}")
print(f"Parameters: c={parent_c}, k={parent_k}")
print(f"Population Size: {sol_per_pop}")
print(f"\nProbability Statistics:")
print(f"  - Best individual (rank 1): {probs_normalized[0]:.6f}")
print(f"  - Median individual (rank {sol_per_pop//2}): {probs_normalized[sol_per_pop//2-1]:.6f}")
print(f"  - Worst individual (rank {sol_per_pop}): {probs_normalized[-1]:.6f}")
print(f"  - Ratio (best/worst): {probs_normalized[0]/probs_normalized[-1]:.2f}x")
print(f"\nCumulative Probabilities:")
print(f"  - Top 10% get: {100*np.sum(probs_normalized[:max(1, sol_per_pop//10)]):.2f}% of selections")
print(f"  - Top 25% get: {100*np.sum(probs_normalized[:max(1, sol_per_pop//4)]):.2f}% of selections")
print(f"  - Top 50% get: {100*np.sum(probs_normalized[:sol_per_pop//2]):.2f}% of selections")
print(f"{'='*60}\n")
