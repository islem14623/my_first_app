"""
generate_figures.py
Generate publication-quality figures for the PFE memoir
Author: CHENAFI Islem & BOURAKBA Redha Anouar
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create folder for figures
os.makedirs('figures', exist_ok=True)

# Set style for all figures
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12

# ====================================================================
# YOUR REAL DATA (from your experiments)
# ====================================================================

class_names = ['Normal', 'Attack']

# Confusion matrices (real values from your experiments)
cm_baseline = np.array([[321193, 1936],    # [TN, FP]
                         [1101, 119611]])    # [FN, TP]

cm_ga = np.array([[321023, 2106],
                   [6832, 113880]])

cm_pso = np.array([[321059, 2070],
                    [5816, 114896]])

cm_aco = np.array([[319912, 3217],
                    [5207, 115505]])

# Results data
models = ['RF\n(Baseline)', 'GA + CNN', 'PSO + CNN', 'ACO + CNN']
accuracy = [99.32, 97.99, 98.22, 98.10]
precision = [98.41, 98.18, 98.23, 97.29]
recall = [99.09, 94.34, 95.18, 95.69]
f1_score = [98.75, 96.22, 96.68, 96.48]
features = [20, 18, 23, 20]

# Colors (professional palette)
colors_models = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

# ====================================================================
# FUNCTION: Create confusion matrix figure
# ====================================================================

def plot_confusion_matrix(cm, title, filename, cmap='Blues'):
    """Create a beautiful confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(7, 6))
    
    # Create heatmap
    sns.heatmap(cm, annot=True, fmt=',', cmap=cmap, cbar=True,
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={"size": 14, "weight": "bold"},
                square=True, linewidths=2, linecolor='white',
                ax=ax)
    
    # Title and labels
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('Actual Label', fontsize=12, fontweight='bold')
    
    # Make tick labels bold
    plt.xticks(fontsize=11, fontweight='bold')
    plt.yticks(fontsize=11, fontweight='bold', rotation=0)
    
    plt.tight_layout()
    plt.savefig(f'figures/{filename}', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: figures/{filename}")

# ====================================================================
# FIGURE 1-4: Confusion matrices
# ====================================================================

print("\n[1] Generating confusion matrices...")

plot_confusion_matrix(cm_baseline, 
                      'Random Forest Baseline\nAccuracy: 99.32%',
                      'confusion_matrix_baseline.png',
                      cmap='Blues')

plot_confusion_matrix(cm_ga, 
                      'GA + CNN\nAccuracy: 97.99%',
                      'confusion_matrix_ga.png',
                      cmap='Greens')

plot_confusion_matrix(cm_pso, 
                      'PSO + CNN\nAccuracy: 98.22%',
                      'confusion_matrix_pso.png',
                      cmap='Oranges')

plot_confusion_matrix(cm_aco, 
                      'ACO + CNN\nAccuracy: 98.10%',
                      'confusion_matrix_aco.png',
                      cmap='Reds')

# ====================================================================
# FIGURE 5: Accuracy Comparison Bar Chart
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

# Highlight the best
bars[0].set_edgecolor('gold')
bars[0].set_linewidth(3)

ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
ax.set_title('Accuracy Comparison of All Models', 
             fontsize=15, fontweight='bold', pad=15)
ax.set_ylim(96, 100)
ax.grid(axis='y', linestyle='--', alpha=0.5)
ax.set_axisbelow(True)

plt.xticks(fontsize=12, fontweight='bold')
plt.yticks(fontsize=11)
plt.tight_layout()
plt.savefig('figures/accuracy_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: figures/accuracy_comparison.png")

# ====================================================================
# FIGURE 6: Features Comparison Bar Chart
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

# Add a horizontal line at 42 (total features)
ax.axhline(y=42, color='red', linestyle='--', linewidth=2, 
           label='Total: 42 features')

# Highlight smallest model
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
# FIGURE 7: All Metrics Comparison (bonus!)
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
ax.set_ylim(90, 101)
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
print("✓ ALL FIGURES GENERATED SUCCESSFULLY!")
print("="*60)
print(f"\nFigures saved in: ./figures/")
print("\nFiles created:")
print("  1. confusion_matrix_baseline.png")
print("  2. confusion_matrix_ga.png")
print("  3. confusion_matrix_pso.png")
print("  4. confusion_matrix_aco.png")
print("  5. accuracy_comparison.png")
print("  6. features_comparison.png")
print("  7. all_metrics_comparison.png  (bonus)")
print("\nYou can now insert these in your memoir!")
