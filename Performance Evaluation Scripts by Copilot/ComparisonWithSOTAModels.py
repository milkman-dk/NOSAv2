import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def create_visual_comparison():
    """Create the most visual comparison of model performance"""
    
    # Data for comparison - only models with both metrics
    models_data = {
        'Model': [
            'Enhanced U-Net',
            'EfficientNetB4+Multi-Attention U-Net',
            'ARU-Net',
            'NOSA v2',
            'Baseline U-Net'
        ],
        'Accuracy': [99.8, 99.8, 98.3, 98.153, 93.9],
        'IoU': [91.3, 87.9, 96.3, 32.1, 86.6],
        'Type': ['Literature', 'Literature', 'Literature', 'Your Model', 'Literature'],
        'Year': [2022, 2025, 2025, 2025, 2025]
    }
    
    df = pd.DataFrame(models_data)
    
    # Create figure with custom styling
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Set background color
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#f8f9fa')
    
    # Create scatter plot
    literature_data = df[df['Type'] == 'Literature']
    your_data = df[df['Type'] == 'Your Model']
    
    # Plot literature models
    scatter1 = ax.scatter(literature_data['Accuracy'], literature_data['IoU'], 
                         s=400, alpha=0.8, c='#2E86AB', marker='o', 
                         edgecolors='white', linewidth=3, label='State-of-the-Art Models',
                         zorder=5)
    
    # Plot your model with special highlighting
    scatter2 = ax.scatter(your_data['Accuracy'], your_data['IoU'], 
                         s=600, alpha=0.9, c='#F24236', marker='s', 
                         edgecolors='white', linewidth=4, label='NOSA v2 (Your Model)',
                         zorder=10)
    
    # Add model names as annotations
    for _, row in literature_data.iterrows():
        ax.annotate(row['Model'], (row['Accuracy'], row['IoU']), 
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=11, fontweight='bold', color='#2E86AB',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                            edgecolor='#2E86AB', alpha=0.8))
    
    for _, row in your_data.iterrows():
        ax.annotate(row['Model'], (row['Accuracy'], row['IoU']), 
                   xytext=(15, 15), textcoords='offset points',
                   fontsize=12, fontweight='bold', color='#F24236',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFE5E5', 
                            edgecolor='#F24236', alpha=0.9, linewidth=2))
    
    # Add performance zones with color coding
    # Excellence zone (high accuracy, high IoU)
    excellence_x = [96, 100, 100, 96]
    excellence_y = [85, 85, 100, 100]
    ax.fill(excellence_x, excellence_y, color='#90EE90', alpha=0.2, label='Excellence Zone')
    
    # Good zone (medium-high performance)
    good_x = [94, 96, 96, 94]
    good_y = [70, 70, 85, 85]
    ax.fill(good_x, good_y, color='#FFD700', alpha=0.2, label='Good Zone')
    
    # Add grid with custom styling
    ax.grid(True, linestyle='--', alpha=0.4, color='gray')
    ax.set_axisbelow(True)
    
    # Customize axes
    ax.set_xlabel('Accuracy (%)', fontsize=16, fontweight='bold', color='#333333')
    ax.set_ylabel('IoU (%)', fontsize=16, fontweight='bold', color='#333333')
    ax.set_title('Brain Tumor Segmentation Models:\nAccuracy vs IoU Performance Map', 
                fontsize=18, fontweight='bold', color='#333333', pad=25)
    
    # Set axis limits with padding
    ax.set_xlim(92, 101)
    ax.set_ylim(25, 100)
    
    # Add reference lines
    ax.axhline(y=90, color='red', linestyle=':', alpha=0.6, linewidth=2)
    ax.axvline(x=98, color='red', linestyle=':', alpha=0.6, linewidth=2)
    
    # Add text annotations for reference lines
    ax.text(92.5, 91, 'IoU Target: 90%', fontsize=10, color='red', fontweight='bold')
    ax.text(98.2, 27, 'Accuracy Target: 98%', fontsize=10, color='red', 
           fontweight='bold', rotation=90)
    
    # Customize legend
    legend = ax.legend(loc='lower left', fontsize=12, frameon=True, 
                      fancybox=True, shadow=True, framealpha=0.9)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('gray')
    
    # Tight layout and save
    plt.tight_layout()
    plt.savefig('visual_model_comparison.png', dpi=300, bbox_inches='tight', 
               facecolor='white', edgecolor='none')
    plt.show()
    
    # Create a complementary bar chart comparison
    create_bar_comparison(df)

def create_bar_comparison(df):
    """Create a complementary bar chart for clear metric comparison"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('Model Performance Breakdown', fontsize=18, fontweight='bold')
    
    # Colors
    colors = ['#2E86AB' if model != 'NOSA v2' else '#F24236' for model in df['Model']]
    
    # Accuracy comparison
    bars1 = ax1.bar(range(len(df)), df['Accuracy'], color=colors, alpha=0.8, 
                    edgecolor='white', linewidth=2)
    ax1.set_title('Accuracy Comparison', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Accuracy (%)', fontsize=14)
    ax1.set_ylim(90, 101)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars1, df['Accuracy']):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                f'{val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # IoU comparison
    bars2 = ax2.bar(range(len(df)), df['IoU'], color=colors, alpha=0.8, 
                    edgecolor='white', linewidth=2)
    ax2.set_title('IoU Comparison', fontsize=16, fontweight='bold')
    ax2.set_ylabel('IoU (%)', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars2, df['IoU']):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # Set x-axis labels
    for ax in [ax1, ax2]:
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(df['Model'], rotation=45, ha='right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('bar_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    create_visual_comparison()
    print("\nGraphs saved as:")
    print("• visual_model_comparison.png (main scatter plot)")
    print("• bar_comparison.png (bar chart comparison)")