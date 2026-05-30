"""
generate_figures.py - UPDATED WITH NEW REAL RESULTS
Generate publication-quality figures for the PFE memoir
Author: CHENAFI Islem & BOURAKBA Redha Anouar
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create folder for figures
os.makedirs('figures', exist_ok=True)

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12

# ====================================================================
# UPDATED REAL DATA (from your latest experiments)
# ====================================================================

class_names = ['Normal', 'Attack']

# Confusion matrices (NEW REAL VALUES)
cm_baseline = np.array([[321201, 1928],    # [TN, FP]
                         [1113, 119599]])    # [FN, TP]

cm_ga = np.array([[319284, 3845],
                   [12747, 107965]])

cm_pso = np.array([[321700, 1429],
                    [8016, 112696]])

cm_aco = np.array([[322255, 874],
                    [7826, 112886]])

# UPDATED Results data
models = ['RF\n(Baseline)', 'GA + CNN', 'PSO + CNN', 'ACO + CNN']
accuracy = [99.31, 96.26, 97.87, 98.04]
precision = [98.41, 96.56, 98.75, 99.23]
recall = [99.08, 89.44, 93.36, 93.52]
f1_score = [98.74, 92.86, 95.98, 96.29]
features = [20, 21, 25, 20]

# Colors (professional palette)
colors_models = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

# ====================================================================
# Helper function for confusion matrix
# ====================================================================

def plot_confusion_matrix(cm, title, filename, cmap='Blues'):
    """Create a beautiful confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(7, 6))
    
    sns.heatmap(cm, annot=True, fmt=',', cmap=cmap, cbar=True,
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={"size": 14, "weight": "bold"},
                square=True, linewidths=2, linecolor='white',
                ax=ax)
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('Actual Label', fontsize=12, fontweight='bold')
    
    plt.xticks(fontsize=11, fontweight='bold')
    plt.yticks(fontsize=11, fontweight='bold', rotation=0)
    
    plt.tight_layout()
    plt.savefig(f'figures/{filename}', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: figures/{filename}")

# ====================================================================
# FIGURES 1-4: Confusion matrices (UPDATED)
# ====================================================================

print("\n[1] Generating confusion matrices with NEW numbers...")

plot_confusion_matrix(cm_baseline, 
                      'Random Forest Baseline\nAccuracy: 99.31%',
                      'confusion_matrix_baseline.png',
                      cmap='Blues')

plot_confusion_matrix(cm_ga, 
                      'GA + CNN\nAccuracy: 96.26%',
                      'confusion_matrix_ga.png',
                      cmap='Greens')

plot_confusion_matrix(cm_pso, 
                      'PSO + CNN\nAccuracy: 97.87%',
                      'confusion_matrix_pso.png',
                      cmap='Oranges')

plot_confusion_matrix(cm_aco, 
                      'ACO + CNN\nAccuracy: 98.04%',
                      'confusion_matrix_aco.png',
                      cmap='Reds')

# ====================================================================
# FIGURE 5: Accuracy Comparison Bar Chart (UPDATED)
# ====================================================================

print("\n[2] Generating accuracy comparison chart...")

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(models, accuracy, color=colors_models, 
              edgecolor='black', linewidth=1.5, width=0.6)

# Add value labels on top of bars
for bar, value in zip(bars, accuracy):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
            f'{value}%', ha='center', va='bottom', 
            fontsize=13, fontweight='bold')

# Highlight the best (baseline)
bars[0].set_edgecolor('gold')
bars[0].set_linewidth(3)

ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
ax.set_title('Accuracy Comparison of All Models', 
             fontsize=15, fontweight='bold', pad=15)
ax.set_ylim(94, 100)
ax.grid(axis='y', linestyle='--', alpha=0.5)
ax.set_axisbelow(True)

plt.xticks(fontsize=12, fontweight='bold')
plt.yticks(fontsize=11)
plt.tight_layout()
plt.savefig('figures/accuracy_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: figures/accuracy_comparison.png")

# ====================================================================
# FIGURE 6: Features Comparison Bar Chart (UPDATED)
# ====================================================================

print("\n[3] Generating features comparison chart...")

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(models, features, color=colors_models, 
              edgecolor='black', linewidth=1.5, width=0.6)

# Add value labels
for bar, value in zip(bars, features):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'{value} / 42', ha='center', va='bottom', 
            fontsize=13, fontweight='bold')

# Add horizontal line at 42 (total features)
ax.axhline(y=42, color='red', linestyle='--', linewidth=2, 
           label='Total: 42 features')

# Highlight smallest model (GA)
min_idx = features.index(min(features))
bars[min_idx].set_edgecolor('gold')
bars[min_idx].set_linewidth(3)

ax.set_ylabel('Number of Selected Features', fontsize=13, fontweight='bold')
ax.set_title('Feature Selection Comparison', 
             fontsize=15, fontweight='bold', pad=15)
ax.set_ylim(0, 50)
ax.grid(axis='y', linestyle='--', alpha=0.5)
ax.set_axisbelow(True)
ax.legend(loc='upper right', fontsize=11)

plt.xticks(fontsize=12, fontweight='bold')
plt.yticks(fontsize=11)
plt.tight_layout()
plt.savefig('figures/features_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: figures/features_comparison.png")

# ====================================================================
# FIGURE 7: All Metrics Comparison (UPDATED)
# ====================================================================

print("\n[4] Generating all metrics comparison chart...")

x = np.arange(len(models))
width = 0.2

fig, ax = plt.subplots(figsize=(12, 7))

bars1 = ax.bar(x - 1.5*width, accuracy, width, label='Accuracy', 
                color='#2E86AB', edgecolor='black')
bars2 = ax.bar(x - 0.5*width, precision, width, label='Precision', 
                color='#A23B72', edgecolor='black')
bars3 = ax.bar(x + 0.5*width, recall, width, label='Recall', 
                color='#F18F01', edgecolor='black')
bars4 = ax.bar(x + 1.5*width, f1_score, width, label='F1-Score', 
                color='#C73E1D', edgecolor='black')

ax.set_xlabel('Models', fontsize=13, fontweight='bold')
ax.set_ylabel('Score (%)', fontsize=13, fontweight='bold')
ax.set_title('Performance Comparison of All Models (Attack Class)', 
             fontsize=15, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=11, fontweight='bold')
ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
ax.set_ylim(85, 101)
ax.grid(axis='y', linestyle='--', alpha=0.5)
ax.set_axisbelow(True)

# Add value labels
for bars in [bars1, bars2, bars3, bars4]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8, rotation=0)

plt.tight_layout()
plt.savefig('figures/all_metrics_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: figures/all_metrics_comparison.png")

# ====================================================================
# DONE
# ====================================================================

print("\n" + "="*60)
print("✓ ALL FIGURES REGENERATED WITH NEW REAL RESULTS!")
print("="*60)
print(f"\nFigures saved in: ./figures/")
print("\nFiles updated:")
print("  1. confusion_matrix_baseline.png  (99.31%)")
print("  2. confusion_matrix_ga.png         (96.26%)")
print("  3. confusion_matrix_pso.png        (97.87%)")
print("  4. confusion_matrix_aco.png        (98.04%)")
print("  5. accuracy_comparison.png")
print("  6. features_comparison.png")
print("  7. all_metrics_comparison.png")
print("\n✓ All numbers reflect your REAL terminal outputs!")
print("✓ You can now upload them to Overleaf!")
