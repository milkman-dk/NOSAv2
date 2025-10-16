import os
import sys
import numpy as np
import mat73
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add paths for both NOSA versions
sys.path.append('NOSAv2')
sys.path.append('NOSA')

# Import both versions
from NOSAv2.NOSA_v2 import predict_tumor as predict_v2
from NOSA.NOSA_v1 import predict_tumor as predict_v1

class SegmentationMetrics:
    """Class to calculate various segmentation performance metrics"""
    
    @staticmethod
    def jaccard_index(y_true, y_pred):
        """Calculate Jaccard Index (IoU)"""
        intersection = np.sum(y_true * y_pred)
        union = np.sum(y_true) + np.sum(y_pred) - intersection
        return intersection / (union + 1e-8)
    
    @staticmethod
    def sensitivity_specificity(y_true, y_pred):
        """Calculate Sensitivity (Recall) and Specificity"""
        tn, fp, fn, tp = confusion_matrix(y_true.flatten(), y_pred.flatten()).ravel()
        sensitivity = tp / (tp + fn + 1e-8)  # True Positive Rate
        specificity = tn / (tn + fp + 1e-8)  # True Negative Rate
        return sensitivity, specificity
    
    @staticmethod
    def precision_recall_f1(y_true, y_pred):
        """Calculate Precision, Recall, and F1 score"""
        tn, fp, fn, tp = confusion_matrix(y_true.flatten(), y_pred.flatten()).ravel()
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        return precision, recall, f1
    
    @staticmethod
    def hausdorff_distance(y_true, y_pred):
        """Simplified Hausdorff distance calculation"""
        from scipy.spatial.distance import directed_hausdorff
        
        # Get boundary points
        true_points = np.argwhere(y_true == 1)
        pred_points = np.argwhere(y_pred == 1)
        
        if len(true_points) == 0 or len(pred_points) == 0:
            return float('inf')
        
        hd1 = directed_hausdorff(true_points, pred_points)[0]
        hd2 = directed_hausdorff(pred_points, true_points)[0]
        return max(hd1, hd2)

class ModelComparison:
    """Class to compare NOSA v1 and v2 performance"""
    
    def __init__(self, test_data_dir, results_dir="performance_results"):
        self.test_data_dir = test_data_dir
        self.results_dir = results_dir
        self.metrics = SegmentationMetrics()
        
        # Create results directory
        os.makedirs(results_dir, exist_ok=True)
        
        # Initialize results storage
        self.results = {
            'NOSAv1': [],
            'NOSAv2': []
        }
        
    def load_test_files(self):
        """Load all .mat files from test directory"""
        test_files = []
        for file in os.listdir(self.test_data_dir):
            if file.endswith('.mat'):
                file_path = os.path.join(self.test_data_dir, file)
                test_files.append(file_path)
        return test_files
    
    def load_ground_truth(self, file_path):
        """Load ground truth mask from .mat file"""
        try:
            mat = mat73.loadmat(file_path)
            if 'cjdata' in mat and 'tumorMask' in mat['cjdata']:
                return np.array(mat['cjdata']['tumorMask'], dtype=np.uint8)
            else:
                return None
        except:
            return None
    
    def evaluate_single_image(self, file_path, model_version):
        """Evaluate a single image with specified model version"""
        try:
            # Load ground truth
            gt_mask = self.load_ground_truth(file_path)
            if gt_mask is None:
                return None
            
            # Get prediction based on model version
            if model_version == 'v1':
                pred_mask, _ = predict_v1(file_path)
            else:  # v2
                pred_mask, _ = predict_v2(file_path)
            
            # Ensure same dimensions
            if gt_mask.shape != pred_mask.shape:
                # Resize prediction to match ground truth
                from skimage.transform import resize
                pred_mask = resize(pred_mask, gt_mask.shape, preserve_range=True, anti_aliasing=False)
                pred_mask = (pred_mask > 0.5).astype(np.uint8)
            
            # Calculate metrics
            metrics_dict = {
                'file': os.path.basename(file_path),
                'jaccard': self.metrics.jaccard_index(gt_mask, pred_mask),
                'precision': 0,
                'recall': 0,
                'f1': 0,
                'specificity': 0,
                'hausdorff': 0
            }
            
            # Calculate additional metrics
            precision, recall, f1 = self.metrics.precision_recall_f1(gt_mask, pred_mask)
            sensitivity, specificity = self.metrics.sensitivity_specificity(gt_mask, pred_mask)
            
            metrics_dict.update({
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'specificity': specificity
            })
            
            # Hausdorff distance (computationally expensive, skip for large datasets)
            try:
                hd = self.metrics.hausdorff_distance(gt_mask, pred_mask)
                metrics_dict['hausdorff'] = hd if hd != float('inf') else 999
            except:
                metrics_dict['hausdorff'] = 999
            
            return metrics_dict
            
        except Exception as e:
            print(f"Error evaluating {file_path}: {str(e)}")
            return None
    
    def run_comparison(self):
        """Run full comparison between NOSA v1 and v2"""
        test_files = self.load_test_files()
        
        if not test_files:
            print("No test files found!")
            return
        
        print(f"Found {len(test_files)} test files")
        
        # Evaluate NOSAv1
        print("\nEvaluating NOSA v1...")
        for file_path in tqdm(test_files, desc="NOSA v1"):
            result = self.evaluate_single_image(file_path, 'v1')
            if result:
                self.results['NOSAv1'].append(result)
        
        # Evaluate NOSAv2
        print("\nEvaluating NOSA v2...")
        for file_path in tqdm(test_files, desc="NOSA v2"):
            result = self.evaluate_single_image(file_path, 'v2')
            if result:
                self.results['NOSAv2'].append(result)
        
        print(f"\nSuccessfully evaluated {len(self.results['NOSAv1'])} files with NOSA v1")
        print(f"Successfully evaluated {len(self.results['NOSAv2'])} files with NOSA v2")
    
    def calculate_summary_statistics(self):
        """Calculate summary statistics for both models"""
        summary = {}
        
        for model_name, results in self.results.items():
            if not results:
                continue
                
            df = pd.DataFrame(results)
            
            summary[model_name] = {
                'mean_jaccard': df['jaccard'].mean(),
                'std_jaccard': df['jaccard'].std(),
                'mean_f1': df['f1'].mean(),
                'std_f1': df['f1'].std(),
                'mean_precision': df['precision'].mean(),
                'std_precision': df['precision'].std(),
                'mean_recall': df['recall'].mean(),
                'std_recall': df['recall'].std(),
                'mean_specificity': df['specificity'].mean(),
                'std_specificity': df['specificity'].std(),
                'median_hausdorff': df['hausdorff'].median(),
                'sample_count': len(results)
            }
        
        return summary
    
    def create_visualizations(self):
        """Create comparison visualizations"""
        
        # Convert results to DataFrames
        df_v1 = pd.DataFrame(self.results['NOSAv1'])
        df_v2 = pd.DataFrame(self.results['NOSAv2'])
        
        if df_v1.empty or df_v2.empty:
            print("No data to visualize!")
            return
        
        # Add model version column
        df_v1['model'] = 'NOSA v1'
        df_v2['model'] = 'NOSA v2'
        
        # Combine DataFrames
        df_combined = pd.concat([df_v1, df_v2], ignore_index=True)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('NOSA v1 vs v2 Performance Comparison on Test Data', fontsize=16, fontweight='bold')
        
        # Metrics to plot (removed dice and sensitivity as they duplicate f1 and recall)
        metrics = ['jaccard', 'f1', 'precision', 'recall']
        
        for i, metric in enumerate(metrics):
            row = i // 2
            col = i % 2
            
            # Box plot comparison
            sns.boxplot(data=df_combined, x='model', y=metric, ax=axes[row, col])
            axes[row, col].set_title(f'{metric.capitalize()} Score Comparison')
            axes[row, col].set_ylabel(f'{metric.capitalize()} Score')
            
            # Add mean values as text
            v1_mean = df_v1[metric].mean()
            v2_mean = df_v2[metric].mean()
            axes[row, col].text(0, v1_mean + 0.02, f'μ={v1_mean:.3f}', ha='center', fontweight='bold')
            axes[row, col].text(1, v2_mean + 0.02, f'μ={v2_mean:.3f}', ha='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'model_comparison_boxplots.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        # Create line plot for mean values comparison
        plt.figure(figsize=(10, 6))
        
        # Calculate means for both models
        v1_means = [df_v1[metric].mean() for metric in metrics]
        v2_means = [df_v2[metric].mean() for metric in metrics]
        
        # Create the line plot
        plt.plot(metrics, v1_means, marker='o', linewidth=3, markersize=8, 
                 color='blue', label='NOSA v1', markerfacecolor='lightblue', 
                 markeredgecolor='blue', markeredgewidth=2)
        plt.plot(metrics, v2_means, marker='s', linewidth=3, markersize=8, 
                 color='red', label='NOSA v2', markerfacecolor='lightcoral', 
                 markeredgecolor='red', markeredgewidth=2)
        
        # Add value labels on points
        for i, (metric, v1_val, v2_val) in enumerate(zip(metrics, v1_means, v2_means)):
            plt.text(i, v1_val + 0.02, f'{v1_val:.3f}', ha='center', va='bottom', 
                    fontweight='bold', color='blue', fontsize=10)
            plt.text(i, v2_val - 0.03, f'{v2_val:.3f}', ha='center', va='top', 
                    fontweight='bold', color='red', fontsize=10)
        
        plt.xlabel('Metrics', fontsize=12, fontweight='bold')
        plt.ylabel('Mean Score', fontsize=12, fontweight='bold')
        plt.title('Performance Comparison: NOSA v1 vs v2 on Test Data', fontsize=14, fontweight='bold')
        plt.xticks(range(len(metrics)), [metric.capitalize() for metric in metrics])
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1.1)
        
        # Add improvement annotations
        for i, (v1_val, v2_val) in enumerate(zip(v1_means, v2_means)):
            improvement = ((v2_val - v1_val) / v1_val) * 100 if v1_val > 0 else 0
            if improvement > 0:
                plt.annotate(f'+{improvement:.1f}%', xy=(i, max(v1_val, v2_val) + 0.05), 
                            ha='center', fontsize=9, color='green', fontweight='bold')
            elif improvement < 0:
                plt.annotate(f'{improvement:.1f}%', xy=(i, max(v1_val, v2_val) + 0.05), 
                            ha='center', fontsize=9, color='orange', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'model_comparison_line_graph.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_results(self):
        """Save detailed results to CSV files"""
        
        # Save individual results
        if self.results['NOSAv1']:
            df_v1 = pd.DataFrame(self.results['NOSAv1'])
            df_v1.to_csv(os.path.join(self.results_dir, 'NOSAv1_detailed_results.csv'), index=False)
        
        if self.results['NOSAv2']:
            df_v2 = pd.DataFrame(self.results['NOSAv2'])
            df_v2.to_csv(os.path.join(self.results_dir, 'NOSAv2_detailed_results.csv'), index=False)
        
        # Save summary statistics
        summary = self.calculate_summary_statistics()
        summary_df = pd.DataFrame(summary).T
        summary_df.to_csv(os.path.join(self.results_dir, 'summary_statistics.csv'))
        
        print(f"\nResults saved to {self.results_dir}/")
        return summary
    
    def print_summary_report(self):
        """Print a formatted summary report"""
        summary = self.calculate_summary_statistics()
        
        print("\n" + "="*60)
        print("NOSA MODEL COMPARISON SUMMARY REPORT")
        print("="*60)
        
        if 'NOSAv1' in summary and 'NOSAv2' in summary:
            print(f"\nSample Size: {summary['NOSAv1']['sample_count']} images")
            print("\nMETRIC COMPARISON:")
            print("-" * 60)
            print(f"{'Metric':<15} {'NOSA v1':<20} {'NOSA v2':<20} {'Improvement':<10}")
            print("-" * 60)
            
            # Updated metrics list (removed dice and sensitivity)
            metrics = ['jaccard', 'f1', 'precision', 'recall', 'specificity']
            
            for metric in metrics:
                v1_val = summary['NOSAv1'][f'mean_{metric}']
                v2_val = summary['NOSAv2'][f'mean_{metric}']
                improvement = ((v2_val - v1_val) / v1_val) * 100 if v1_val > 0 else 0
                
                print(f"{metric.capitalize():<15} {v1_val:.3f} ± {summary['NOSAv1'][f'std_{metric}']:8.3f} "
                      f"{v2_val:.3f} ± {summary['NOSAv2'][f'std_{metric}']:8.3f} "
                      f"{improvement:+.1f}%")
            
            print("-" * 60)
            
            # Statistical significance test (using F1 instead of dice)
            f1_v1 = [r['f1'] for r in self.results['NOSAv1']]
            f1_v2 = [r['f1'] for r in self.results['NOSAv2']]
            
            from scipy import stats
            t_stat, p_value = stats.ttest_rel(f1_v2, f1_v1)
            
            print(f"\nSTATISTICAL ANALYSIS:")
            print(f"Paired t-test (F1 coefficient): t={t_stat:.3f}, p={p_value:.4f}")
            if p_value < 0.05:
                print("*** Statistically significant difference (p < 0.05)")
            else:
                print("No statistically significant difference (p >= 0.05)")
                
        print("\n" + "="*60)

def main():
    """Main function to run the comparison"""
    
    # Configuration
    TEST_DATA_DIR = "C:/Matura/testing"  # Update this path to your test data directory
    RESULTS_DIR = "performance_results"
    
    # Check if test directory exists
    if not os.path.exists(TEST_DATA_DIR):
        print(f"Test data directory not found: {TEST_DATA_DIR}")
        print("Please update the TEST_DATA_DIR variable to point to your test data.")
        return
    
    # Initialize comparison
    comparison = ModelComparison(TEST_DATA_DIR, RESULTS_DIR)
    
    print("Starting NOSA v1 vs v2 Performance Comparison...")
    print(f"Test data directory: {TEST_DATA_DIR}")
    print(f"Results will be saved to: {RESULTS_DIR}")
    
    # Run the comparison
    comparison.run_comparison()
    
    # Generate and save results
    summary = comparison.save_results()
    
    # Create visualizations
    comparison.create_visualizations()
    
    # Print summary report
    comparison.print_summary_report()
    
    print(f"\nComparison complete! Check {RESULTS_DIR}/ for detailed results and visualizations.")

if __name__ == "__main__":
    main()