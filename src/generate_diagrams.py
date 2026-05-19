"""
generate_diagrams.py
Generate all architecture and flowchart diagrams for the PFE memoir
Author: CHENAFI Islem & BOURAKBA Redha Anouar
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D
import numpy as np
import os

# Create folder
os.makedirs('figures', exist_ok=True)

# Common settings
plt.rcParams['font.family'] = 'serif'

# Color palettes
COLOR_BLUE = '#3498DB'
COLOR_GREEN = '#27AE60'
COLOR_ORANGE = '#E67E22'
COLOR_RED = '#E74C3C'
COLOR_PURPLE = '#9B59B6'
COLOR_GRAY = '#95A5A6'
COLOR_DARK = '#2C3E50'

# ====================================================================
# Helper function: Create a box with text
# ====================================================================

def draw_box(ax, x, y, width, height, text, color, text_color='white', 
             fontsize=11, fontweight='bold', alpha=1.0):
    box = FancyBboxPatch((x, y), width, height,
                          boxstyle="round,pad=0.05",
                          facecolor=color, edgecolor='black',
                          linewidth=2, alpha=alpha)
    ax.add_patch(box)
    ax.text(x + width/2, y + height/2, text,
            ha='center', va='center',
            fontsize=fontsize, fontweight=fontweight,
            color=text_color, wrap=True)

def draw_arrow(ax, x1, y1, x2, y2, color='black', linewidth=2):
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                             arrowstyle='->,head_width=0.4,head_length=0.6',
                             color=color, linewidth=linewidth,
                             mutation_scale=15)
    ax.add_patch(arrow)

# ====================================================================
# FIGURE 1.1: IoT 3-Layer Architecture
# ====================================================================

print("[1] Generating Figure 1.1: IoT 3-Layer Architecture...")

fig, ax = plt.subplots(figsize=(10, 7))

# Layers from top to bottom
draw_box(ax, 1, 5.5, 8, 1.2, 
         'APPLICATION LAYER\n(Smart Home, Healthcare, Smart Cities)', 
         COLOR_BLUE, fontsize=12)

draw_box(ax, 1, 3.5, 8, 1.2, 
         'NETWORK LAYER\n(Wi-Fi, Bluetooth, ZigBee, 4G/5G)', 
         COLOR_GREEN, fontsize=12)

draw_box(ax, 1, 1.5, 8, 1.2, 
         'PERCEPTION LAYER\n(Sensors, RFID, Cameras, GPS)', 
         COLOR_ORANGE, fontsize=12)

# Arrows
draw_arrow(ax, 5, 5.4, 5, 4.7)
draw_arrow(ax, 5, 3.4, 5, 2.7)

# Labels on arrows
ax.text(5.3, 5.05, 'Data', fontsize=10, style='italic')
ax.text(5.3, 3.05, 'Data', fontsize=10, style='italic')

ax.set_xlim(0, 10)
ax.set_ylim(0.5, 7.5)
ax.axis('off')
ax.set_title('Three-Layer Architecture of IoT', fontsize=14, fontweight='bold', pad=15)

plt.tight_layout()
plt.savefig('figures/iot_architecture.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: figures/iot_architecture.png")

# ====================================================================
# FIGURE 1.2: IIoT 4-Layer Architecture
# ====================================================================

print("[2] Generating Figure 1.2: IIoT 4-Layer Architecture...")

fig, ax = plt.subplots(figsize=(10, 9))

draw_box(ax, 1, 7, 8, 1.2, 
         'CLOUD LAYER\n(Big Data, AI Analytics, Dashboards)', 
         COLOR_PURPLE, fontsize=12)

draw_box(ax, 1, 5, 8, 1.2, 
         'EDGE LAYER\n(Edge Computing, Local Analysis)', 
         COLOR_BLUE, fontsize=12)

draw_box(ax, 1, 3, 8, 1.2, 
         'NETWORK LAYER\n(MQTT, CoAP, Modbus, Profinet)', 
         COLOR_GREEN, fontsize=12)

draw_box(ax, 1, 1, 8, 1.2, 
         'DEVICE LAYER\n(Industrial Sensors, Actuators, PLCs)', 
         COLOR_ORANGE, fontsize=12)

# Arrows (bidirectional)
for y in [6.95, 4.95, 2.95]:
    draw_arrow(ax, 5, y, 5, y - 0.75)

ax.set_xlim(0, 10)
ax.set_ylim(0, 9)
ax.axis('off')
ax.set_title('Four-Layer Architecture of IIoT', fontsize=14, fontweight='bold', pad=15)

plt.tight_layout()
plt.savefig('figures/iiot_architecture.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: figures/iiot_architecture.png")

# ====================================================================
# FIGURE 1.3: HIDS vs NIDS
# ====================================================================

print("[3] Generating Figure 1.3: HIDS vs NIDS Comparison...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))

# === HIDS (left) ===
ax1.set_title('HIDS: Host-Based IDS', fontsize=14, fontweight='bold', color=COLOR_BLUE)

devices_hids = ['Server 1', 'Server 2', 'Server 3']
y_positions = [5, 3.5, 2]
for device, y in zip(devices_hids, y_positions):
    # Device box
    draw_box(ax1, 2, y, 3, 1, device, COLOR_DARK, fontsize=11)
    # Agent label
    draw_box(ax1, 5.2, y + 0.2, 1.5, 0.6, 'IDS\nAgent', COLOR_GREEN, 
             fontsize=9)

ax1.text(4, 0.8, 'IDS installed on EACH device', 
         ha='center', fontsize=11, style='italic', fontweight='bold')

ax1.set_xlim(0, 8)
ax1.set_ylim(0, 7)
ax1.axis('off')

# === NIDS (right) ===
ax2.set_title('NIDS: Network-Based IDS', fontsize=14, fontweight='bold', color=COLOR_ORANGE)

# Router/Firewall at top
draw_box(ax2, 3, 5.5, 4, 1, 'Router / Firewall\n[NIDS Agent]', COLOR_ORANGE, fontsize=11)

# Network devices at bottom
devices_nids = ['PC 1', 'PC 2', 'PC 3']
x_positions = [1, 4, 7]
for device, x in zip(devices_nids, x_positions):
    draw_box(ax2, x - 0.6, 2, 1.5, 0.8, device, COLOR_DARK, fontsize=10)
    # Line from router to each device
    ax2.plot([5, x + 0.15], [5.5, 2.8], 'k-', linewidth=1.5)

ax2.text(5, 0.8, 'ONE IDS monitors ALL traffic', 
         ha='center', fontsize=11, style='italic', fontweight='bold')

ax2.set_xlim(0, 9)
ax2.set_ylim(0, 7)
ax2.axis('off')

plt.suptitle('Comparison between HIDS and NIDS', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('figures/hids_vs_nids.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: figures/hids_vs_nids.png")

# ====================================================================
# FIGURE 1.4: General IDS Architecture
# ====================================================================

print("[4] Generating Figure 1.4: General IDS Architecture...")

fig, ax = plt.subplots(figsize=(14, 4))

components = [
    ('Data\nCollection', COLOR_BLUE),
    ('Preprocessing', COLOR_GREEN),
    ('Feature\nSelection', COLOR_ORANGE),
    ('Detection\nEngine', COLOR_RED),
    ('Alert &\nResponse', COLOR_PURPLE)
]

x_start = 0.5
y = 1.5
width = 2.0
height = 1.5
gap = 0.5

for i, (name, color) in enumerate(components):
    x = x_start + i * (width + gap)
    draw_box(ax, x, y, width, height, name, color, fontsize=11)
    
    # Arrow to next component
    if i < len(components) - 1:
        draw_arrow(ax, x + width, y + height/2, x + width + gap - 0.05, y + height/2)

ax.set_xlim(0, 14)
ax.set_ylim(0, 4)
ax.axis('off')
ax.set_title('General Architecture of an IDS', fontsize=14, fontweight='bold', pad=15)

plt.tight_layout()
plt.savefig('figures/ids_architecture.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: figures/ids_architecture.png")

# ====================================================================
# FIGURE 2.1: Feature Selection Categories
# ====================================================================

print("[5] Generating Figure 2.1: Feature Selection Categories...")

fig, ax = plt.subplots(figsize=(14, 7))

# Title box at top
draw_box(ax, 4.5, 5.5, 5, 1, 'Feature Selection Methods', 
         COLOR_DARK, fontsize=14)

# Three category boxes
categories = [
    ('FILTER', '• Chi-Square\n• Info Gain\n• Correlation\n\nFast, Simple\nNo ML model', COLOR_BLUE, 1),
    ('WRAPPER', '• Genetic Algorithm\n• PSO\n• ACO\n\nSlow, Accurate\nUses ML model', COLOR_GREEN, 5.5),
    ('EMBEDDED', '• LASSO\n• Decision Trees\n• RF Importance\n\nMedium, Good\nDuring training', COLOR_ORANGE, 10)
]

for name, content, color, x in categories:
    # Header
    draw_box(ax, x, 3.5, 3, 0.8, name, color, fontsize=12)
    # Content
    draw_box(ax, x, 0.5, 3, 2.8, content, 'white', text_color='black', 
             fontsize=10, fontweight='normal')

# Arrows from top to each category
for x in [2.5, 7, 11.5]:
    draw_arrow(ax, 7, 5.4, x, 4.4)

ax.set_xlim(0, 14)
ax.set_ylim(0, 7)
ax.axis('off')
ax.set_title('Three Main Categories of Feature Selection Methods', 
             fontsize=14, fontweight='bold', pad=15)

plt.tight_layout()
plt.savefig('figures/fs_categories.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: figures/fs_categories.png")

# ====================================================================
# Helper function for flowcharts
# ====================================================================

def draw_flowchart_box(ax, x, y, w, h, text, color, fontsize=10):
    box = FancyBboxPatch((x, y), w, h,
                          boxstyle="round,pad=0.05",
                          facecolor=color, edgecolor='black',
                          linewidth=2)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text,
            ha='center', va='center',
            fontsize=fontsize, fontweight='bold', color='white')

def draw_diamond(ax, x, y, w, h, text, color):
    diamond = plt.Polygon([(x + w/2, y + h), (x + w, y + h/2),
                            (x + w/2, y), (x, y + h/2)],
                           facecolor=color, edgecolor='black', linewidth=2)
    ax.add_patch(diamond)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=9, fontweight='bold', color='white')

# ====================================================================
# FIGURE 2.2: GA Flowchart
# ====================================================================

print("[6] Generating Figure 2.2: GA Flowchart...")

fig, ax = plt.subplots(figsize=(8, 12))

steps_ga = [
    (3, 11, 2, 0.7, 'START', COLOR_RED),
    (2.5, 9.5, 3, 0.8, 'Initialize Population', COLOR_BLUE),
    (2.5, 8, 3, 0.8, 'Evaluate Fitness', COLOR_BLUE),
    (2.5, 6.5, 3, 0.8, 'Selection', COLOR_GREEN),
    (2.5, 5, 3, 0.8, 'Crossover', COLOR_GREEN),
    (2.5, 3.5, 3, 0.8, 'Mutation', COLOR_GREEN),
]

for x, y, w, h, text, color in steps_ga:
    draw_flowchart_box(ax, x, y, w, h, text, color, fontsize=11)

# Decision diamond
draw_diamond(ax, 2.5, 1.8, 3, 1.2, 'Stop?', COLOR_ORANGE)

# END box
draw_flowchart_box(ax, 3, 0.2, 2, 0.7, 'END', COLOR_RED, fontsize=11)

# Arrows
arrows_ga = [(4, 11, 4, 10.3),
              (4, 9.5, 4, 8.8),
              (4, 8, 4, 7.3),
              (4, 6.5, 4, 5.8),
              (4, 5, 4, 4.3),
              (4, 3.5, 4, 3),
              (4, 1.8, 4, 0.9)]

for x1, y1, x2, y2 in arrows_ga:
    draw_arrow(ax, x1, y1, x2, y2)

# Yes/No labels
ax.text(4.3, 1, 'Yes', fontsize=10, fontweight='bold', color=COLOR_RED)

# Loop back arrow (No)
ax.annotate('', xy=(2.5, 7), xytext=(1, 2.4),
            arrowprops=dict(arrowstyle='->', color=COLOR_BLUE, lw=2,
                            connectionstyle="arc3,rad=-0.3"))
ax.text(0.5, 5, 'No', fontsize=10, fontweight='bold', color=COLOR_BLUE)

ax.set_xlim(0, 8)
ax.set_ylim(0, 12)
ax.axis('off')
ax.set_title('Genetic Algorithm Flowchart', fontsize=14, fontweight='bold', pad=15)

plt.tight_layout()
plt.savefig('figures/ga_flowchart.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: figures/ga_flowchart.png")

# ====================================================================
# FIGURE 2.3: PSO Flowchart
# ====================================================================

print("[7] Generating Figure 2.3: PSO Flowchart...")

fig, ax = plt.subplots(figsize=(8, 12))

steps_pso = [
    (3, 11, 2, 0.7, 'START', COLOR_RED),
    (2.5, 9.5, 3, 0.8, 'Initialize Swarm', COLOR_BLUE),
    (2.5, 8, 3, 0.8, 'Evaluate Fitness', COLOR_BLUE),
    (2.5, 6.5, 3, 0.8, 'Update pBest', COLOR_PURPLE),
    (2.5, 5, 3, 0.8, 'Update gBest', COLOR_PURPLE),
    (2.5, 3.5, 3, 0.8, 'Update Velocity\n& Position', COLOR_GREEN),
]

for x, y, w, h, text, color in steps_pso:
    draw_flowchart_box(ax, x, y, w, h, text, color, fontsize=11)

draw_diamond(ax, 2.5, 1.8, 3, 1.2, 'Stop?', COLOR_ORANGE)
draw_flowchart_box(ax, 3, 0.2, 2, 0.7, 'END', COLOR_RED, fontsize=11)

for x1, y1, x2, y2 in arrows_ga:
    draw_arrow(ax, x1, y1, x2, y2)

ax.text(4.3, 1, 'Yes', fontsize=10, fontweight='bold', color=COLOR_RED)

ax.annotate('', xy=(2.5, 7), xytext=(1, 2.4),
            arrowprops=dict(arrowstyle='->', color=COLOR_BLUE, lw=2,
                            connectionstyle="arc3,rad=-0.3"))
ax.text(0.5, 5, 'No', fontsize=10, fontweight='bold', color=COLOR_BLUE)

ax.set_xlim(0, 8)
ax.set_ylim(0, 12)
ax.axis('off')
ax.set_title('Particle Swarm Optimization Flowchart', fontsize=14, fontweight='bold', pad=15)

plt.tight_layout()
plt.savefig('figures/pso_flowchart.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: figures/pso_flowchart.png")

# ====================================================================
# FIGURE 2.4: ACO Flowchart
# ====================================================================

print("[8] Generating Figure 2.4: ACO Flowchart...")

fig, ax = plt.subplots(figsize=(8, 12))

steps_aco = [
    (3, 11, 2, 0.7, 'START', COLOR_RED),
    (2.5, 9.5, 3, 0.8, 'Initialize\nPheromones', COLOR_BLUE),
    (2.5, 8, 3, 0.8, 'Construct\nAnt Solutions', COLOR_PURPLE),
    (2.5, 6.5, 3, 0.8, 'Evaluate Fitness', COLOR_BLUE),
    (2.5, 5, 3, 0.8, 'Evaporation', COLOR_GREEN),
    (2.5, 3.5, 3, 0.8, 'Reinforce\nPheromones', COLOR_GREEN),
]

for x, y, w, h, text, color in steps_aco:
    draw_flowchart_box(ax, x, y, w, h, text, color, fontsize=11)

draw_diamond(ax, 2.5, 1.8, 3, 1.2, 'Stop?', COLOR_ORANGE)
draw_flowchart_box(ax, 3, 0.2, 2, 0.7, 'END', COLOR_RED, fontsize=11)

for x1, y1, x2, y2 in arrows_ga:
    draw_arrow(ax, x1, y1, x2, y2)

ax.text(4.3, 1, 'Yes', fontsize=10, fontweight='bold', color=COLOR_RED)

ax.annotate('', xy=(2.5, 8.4), xytext=(1, 2.4),
            arrowprops=dict(arrowstyle='->', color=COLOR_BLUE, lw=2,
                            connectionstyle="arc3,rad=-0.3"))
ax.text(0.5, 5, 'No', fontsize=10, fontweight='bold', color=COLOR_BLUE)

ax.set_xlim(0, 8)
ax.set_ylim(0, 12)
ax.axis('off')
ax.set_title('Ant Colony Optimization Flowchart', fontsize=14, fontweight='bold', pad=15)

plt.tight_layout()
plt.savefig('figures/aco_flowchart.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: figures/aco_flowchart.png")

# ====================================================================
# FIGURE 3.1: Preprocessing Pipeline
# ====================================================================

print("[9] Generating Figure 3.1: Preprocessing Pipeline...")

fig, ax = plt.subplots(figsize=(16, 4))

steps_pp = [
    ('Load\nCSV', COLOR_BLUE, '2,219,201\nsamples'),
    ('Remove\nNaN', COLOR_BLUE, '2,219,201\nsamples'),
    ('Select\nNumeric', COLOR_GREEN, '42\nfeatures'),
    ('Label\nEncoding', COLOR_GREEN, 'Binary\n(0/1)'),
    ('Feature\nScaling', COLOR_ORANGE, 'Standard\nScaler'),
    ('Train/Test\nSplit', COLOR_RED, '80% / 20%\n1.77M / 443K')
]

x_start = 0.5
y = 1.5
width = 1.8
height = 1.3
gap = 0.6

for i, (name, color, note) in enumerate(steps_pp):
    x = x_start + i * (width + gap)
    draw_box(ax, x, y, width, height, name, color, fontsize=11)
    # Note below
    ax.text(x + width/2, y - 0.4, note, ha='center', va='top',
            fontsize=9, style='italic', color='gray')
    
    if i < len(steps_pp) - 1:
        draw_arrow(ax, x + width, y + height/2, x + width + gap - 0.05, y + height/2)

ax.set_xlim(0, 16)
ax.set_ylim(0, 4)
ax.axis('off')
ax.set_title('Data Preprocessing Pipeline', fontsize=14, fontweight='bold', pad=15)

plt.tight_layout()
plt.savefig('figures/preprocessing_pipeline.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: figures/preprocessing_pipeline.png")

# ====================================================================
# FIGURE 3.2: System Architecture
# ====================================================================

print("[10] Generating Figure 3.2: System Architecture...")

fig, ax = plt.subplots(figsize=(14, 11))

# Dataset at top
draw_box(ax, 4, 9.5, 6, 1, 
         'Edge-IIoTset Dataset\n(2,219,201 samples, 42 features)', 
         COLOR_DARK, fontsize=12)

# Preprocessing
draw_box(ax, 5, 7.8, 4, 0.9, 
         'Preprocessing\n(Clean, Scale, Split)', 
         COLOR_PURPLE, fontsize=11)

# 3 algorithms (parallel)
algorithms = [
    ('GA Feature\nSelection\n(18 features)', COLOR_BLUE, 0.5),
    ('PSO Feature\nSelection\n(23 features)', COLOR_GREEN, 5.5),
    ('ACO Feature\nSelection\n(20 features)', COLOR_ORANGE, 10.5)
]

for name, color, x in algorithms:
    draw_box(ax, x, 5.5, 3, 1.3, name, color, fontsize=11)

# 3 CNNs
for name, color, x in algorithms:
    draw_box(ax, x, 3.2, 3, 1.3, 'CNN Classifier', color, fontsize=11)

# Results
results = [
    ('Accuracy:\n97.99%', COLOR_BLUE, 0.5),
    ('Accuracy:\n98.22%', COLOR_GREEN, 5.5),
    ('Accuracy:\n98.10%', COLOR_ORANGE, 10.5)
]

for name, color, x in results:
    draw_box(ax, x, 1, 3, 1.3, name, color, fontsize=12)

# Arrows
draw_arrow(ax, 7, 9.4, 7, 8.8)
draw_arrow(ax, 7, 7.7, 2, 6.85)
draw_arrow(ax, 7, 7.7, 7, 6.85)
draw_arrow(ax, 7, 7.7, 12, 6.85)

for x in [2, 7, 12]:
    draw_arrow(ax, x, 5.4, x, 4.6)
    draw_arrow(ax, x, 3.1, x, 2.4)

ax.set_xlim(0, 14)
ax.set_ylim(0, 11)
ax.axis('off')
ax.set_title('General Architecture of the Proposed IIoT IDS', 
             fontsize=14, fontweight='bold', pad=15)

plt.tight_layout()
plt.savefig('figures/system_architecture.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: figures/system_architecture.png")

# ====================================================================
# FIGURE 3.3: CNN Architecture
# ====================================================================

print("[11] Generating Figure 3.3: CNN Architecture...")

fig, ax = plt.subplots(figsize=(18, 5))

cnn_layers = [
    ('Input\n(18-23\nfeatures)', COLOR_GRAY),
    ('Conv1D\n(64, k=3)', COLOR_BLUE),
    ('Batch\nNorm', COLOR_PURPLE),
    ('MaxPool\n(2)', COLOR_GREEN),
    ('Conv1D\n(128, k=3)', COLOR_BLUE),
    ('Batch\nNorm', COLOR_PURPLE),
    ('MaxPool\n(2)', COLOR_GREEN),
    ('Flatten', COLOR_ORANGE),
    ('Dense\n(128)', COLOR_RED),
    ('Dropout\n(0.4)', COLOR_GRAY),
    ('Dense\n(64)', COLOR_RED),
    ('Dropout\n(0.3)', COLOR_GRAY),
    ('Sigmoid\n(1)', COLOR_DARK),
]

x_start = 0.2
y = 1.5
width = 1.2
height = 2
gap = 0.2

for i, (name, color) in enumerate(cnn_layers):
    x = x_start + i * (width + gap)
    draw_box(ax, x, y, width, height, name, color, fontsize=9)
    
    if i < len(cnn_layers) - 1:
        draw_arrow(ax, x + width, y + height/2, x + width + gap - 0.02, y + height/2,
                   linewidth=1.5)

# Output label
ax.text(x + width/2, 0.7, 'Normal\nor Attack', 
        ha='center', fontsize=10, fontweight='bold', color=COLOR_DARK,
        style='italic')

ax.set_xlim(0, 18)
ax.set_ylim(0, 5)
ax.axis('off')
ax.set_title('CNN Architecture (1D Convolutional Network)', 
             fontsize=14, fontweight='bold', pad=15)

plt.tight_layout()
plt.savefig('figures/cnn_architecture.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: figures/cnn_architecture.png")

# ====================================================================
# DONE
# ====================================================================

print("\n" + "="*60)
print("✓ ALL 11 DIAGRAMS GENERATED SUCCESSFULLY!")
print("="*60)
print(f"\nFigures saved in: ./figures/")
print("\nNew files created:")
print("  Chapter 1:")
print("    1. iot_architecture.png")
print("    2. iiot_architecture.png")
print("    3. hids_vs_nids.png")
print("    4. ids_architecture.png")
print("  Chapter 2:")
print("    5. fs_categories.png")
print("    6. ga_flowchart.png")
print("    7. pso_flowchart.png")
print("    8. aco_flowchart.png")
print("  Chapter 3:")
print("    9. preprocessing_pipeline.png")
print("   10. system_architecture.png")
print("   11. cnn_architecture.png")
print("\nYou can now insert all these in your memoir!")
