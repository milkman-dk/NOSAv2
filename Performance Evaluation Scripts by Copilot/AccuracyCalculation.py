import os
import sys
import numpy as np
import mat73
import pandas as pd  # Add this line
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
    """Class to calculate segmentation performance metrics"""
    
    @staticmethod
    def accuracy(y_true, y_pred):
        """Calculate accuracy"""
        correct_predictions = np.sum(y_true.flatten() == y_pred.flatten())
        total_predictions = y_true.size
        return correct_predictions / total_predictions * 100  # Return as percentage

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
            
            # Calculate accuracy
            accuracy = self.metrics.accuracy(gt_mask, pred_mask)
            
            # Store results
            metrics_dict = {
                'file': os.path.basename(file_path),
                'accuracy': accuracy
            }
            
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
                'mean_accuracy': df['accuracy'].mean(),
                'std_accuracy': df['accuracy'].std(),
                'sample_count': len(results)
            }
        
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
            
            # Accuracy comparison
            v1_accuracy = summary['NOSAv1']['mean_accuracy']
            v2_accuracy = summary['NOSAv2']['mean_accuracy']
            improvement = ((v2_accuracy - v1_accuracy) / v1_accuracy) * 100 if v1_accuracy > 0 else 0
            
            print(f"{'Accuracy':<15} {v1_accuracy:.3f} ± {summary['NOSAv1']['std_accuracy']:8.3f} "
                  f"{v2_accuracy:.3f} ± {summary['NOSAv2']['std_accuracy']:8.3f} "
                  f"{improvement:+.1f}%")
            
            print("-" * 60)

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
    
    # Print summary report
    comparison.print_summary_report()
    
    print(f"\nComparison complete! Check {RESULTS_DIR}/ for detailed results.")

if __name__ == "__main__":
    main()