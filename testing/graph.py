import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics


class Evaluator:
    """
    A class for evaluating a biometric system's performance.
    """

    def __init__(self, 
                 num_thresholds, 
                 genuine_scores, 
                 impostor_scores, 
                 plot_title, 
                 epsilon=1e-12):
        """
        Initialize the Evaluator object.

        Parameters:
        - num_thresholds (int): Number of thresholds to evaluate.
        - genuine_scores (array-like): Genuine scores for evaluation.
        - impostor_scores (array-like): Impostor scores for evaluation.
        - plot_title (str): Title for the evaluation plots.
        - epsilon (float): A small value to prevent division by zero.
        """
        self.num_thresholds = num_thresholds
        self.thresholds = np.linspace(-0.1, 1.1, num_thresholds)
        self.genuine_scores = genuine_scores
        self.impostor_scores = impostor_scores
        self.plot_title = plot_title
        self.epsilon = epsilon

    def get_dprime(self):
        """
        Calculate the d' (d-prime) metric.
        Returns:
        - float: The calculated d' value.
        """
        x = np.mean(self.genuine_scores) - np.mean(self.impostor_scores)
        y = np.sqrt(np.std(self.genuine_scores) ** 2 + np.std(self.impostor_scores) ** 2) / 2
        return x / (y + self.epsilon)

    def plot_score_distribution(self):
        """
        Plot the distribution of genuine and impostor scores.
        """
        FPR, FNR, _ = self.compute_rates()
        eer_threshold = self.thresholds[np.argmin(np.abs(FPR - FNR))]
        index = np.argmin(np.abs(FPR - FNR))

        plt.figure()
        
        # Plot the histogram for genuine scores
        plt.hist(self.genuine_scores, bins=20, color='green', lw=2, histtype='step', hatch='//', label='Genuine Scores')
        
        # Plot the histogram for impostor scores
        # Provide impostor scores data here
            # color: Set the color for impostor scores
            # lw: Set the line width for the histogram
            # histtype: Choose 'step' for a step histogram
            # hatch: Choose a pattern for filling the histogram bars
            # label: Provide a label for impostor scores in the legend
        plt.hist(self.impostor_scores, bins=20, color='red', lw=2, histtype='step', hatch='.', label='Impostor Scores')
        
        #vertical line for EER threshold and text annotation
        plt.axvline(eer_threshold, color='black', linestyle='--', lw=2)
        plt.text(eer_threshold + 0.05, max(plt.gca().get_ylim()) * 0.8, 
                 f'Score Threshold, t={eer_threshold:.2f}, at EER\nFPR={FPR[index]:.2f}, FNR={FNR[index]:.2f}',
                style='italic', fontsize=10,
                bbox={'facecolor': 'grey', 'alpha': 0.5, 'pad': 5})
        # Set the x-axis limit to ensure the histogram fits within the correct range
        plt.xlim([-0.05, 1.05])
        
        # Add grid lines for better readability
        plt.grid(color='gray', linestyle='--', linewidth=0.5)
        
        # Add legend to the upper left corner with a specified font size
        plt.legend(loc='upper left', fontsize=12)
        
        # Set x and y-axis labels with specified font size and weight
        plt.xlabel("Score", fontsize=12, weight='bold')
        
        plt.ylabel("Frequency", fontsize=12, weight='bold')
        
        # Remove the top and right spines for a cleaner appearance
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        
        # Set font size for x and y-axis ticks
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        
        # Add a title to the plot with d-prime value and system title
        plt.title(f'Score Distribution Plot\nd-prime= {self.get_dprime():.2f}\nSystem {self.plot_title}', fontsize=15, weight='bold')
        
       
        # Save the figure before displaying it
        plt.savefig(f'score_distribution_plot_({self.plot_title}).png', dpi=300, bbox_inches="tight")
        
        # Display the plot after saving
        plt.show()
        
        # Close the figure to free up resources
        plt.close()

        return

    def get_EER(self, FPR, FNR):
        """
        Calculate the Equal Error Rate (EER).
    
        Parameters:
        - FPR (list or array-like): False Positive Rate values.
        - FNR (list or array-like): False Negative Rate values.
    
        Returns:
        - float: Equal Error Rate (EER).
        """
        
        # Add code here to compute the EER
        abs_diff = np.abs(np.array(FPR) - np.array(FNR))
        EER_index = np.argmin(abs_diff)
        EER = (FPR[EER_index] + FNR[EER_index]) / 2
        
        return EER

    def plot_det_curve(self, FPR, FNR):
        """
        Plot the Detection Error Tradeoff (DET) curve.
        Parameters:
         - FPR (list or array-like): False Positive Rate values.
         - FNR (list or array-like): False Negative Rate values.
        """
        
        # Calculate the Equal Error Rate (EER) using the get_EER method
        EER = self.get_EER(FPR, FNR)
        #eer_threshold = self.thresholds[np.argmin(np.abs(FPR - FNR))]
        
        # Create a new figure for plotting
        plt.figure()
        
        # Plot the Detection Error Tradeoff Curve
        plt.plot(FPR, FNR, color='blue', lw=2, label='DET Curve')
        
        # Add a text annotation for the EER point on the curve
        # Plot the diagonal line representing random classification
        # Scatter plot to highlight the EER point on the curve

        plt.text(EER + 0.07, EER + 0.07, f'EER = {EER:.5f}', style='italic', 
                 fontsize=12,
                 bbox={'facecolor': 'grey', 'alpha': 0.5, 'pad': 10})
        plt.plot([0, 1], [0, 1], '--', lw=0.5, color='black')
        plt.scatter([EER], [EER], c="black", s=100, zorder=5)

        # Set the x and y-axis limits to ensure the plot fits within the range 
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])

        # Add grid lines for better readability
        plt.grid(color='gray', linestyle='--', linewidth=0.5)

        # Remove the top and right spines for a cleaner appearance
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)

        # Set x and y-axis labels with specified font size and weight
        plt.xlabel("False Pos. Rate", fontsize=12, weight='bold')
        plt.ylabel("False Neg. Rate", fontsize=12, weight='bold')

        # Add a title to the plot with EER value and system title
        plt.title(f'DET Curve \nEER = {EER:.5f}\nSystem {self.plot_title}', fontsize=15, weight='bold')

        # Set font size for x and y-axis ticks
        plt.xticks(fontsize=12)  
        plt.yticks(fontsize=12)

        # Save the plot as an image file
        plt.savefig(f'ROC_curve_plot_{self.plot_title}.png', dpi=300, bbox_inches="tight")

        # Display the plot
        plt.show() 

        # Close the plot to free up resources
        plt.close()
        return

    def plot_roc_curve(self, FPR, TPR):
        """
        Plot the Receiver Operating Characteristic (ROC) curve.
        Parameters:
        - FPR (list or array-like): False Positive Rate values.
        - TPR (list or array-like): True Positive Rate values.
        """
        
        # Create a new figure for the ROC curve
        plt.figure()
        # Plot the ROC curve using FPR and TPR with specified attributes
        plt.plot(FPR, TPR, lw=2, color='black', label='ROC Curve')
        #plt.plot([0, 1], [0, 1], '--', lw=0.5, color='black')
        
        # Set x and y axis limits, add grid, and remove top and right spines
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.grid(color='gray', linestyle='--', linewidth=0.5)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        
        # Set labels for x and y axes, and add a title
        plt.xlabel('False Positive Rate', fontsize=12, weight='bold')
        plt.ylabel('True Positive Rate', fontsize=12, weight='bold')
        plt.title(f'ROC Curve\nSystem {self.plot_title}', fontsize=15, weight='bold')
        
        # Set font sizes for ticks, x and y labels
        plt.xticks(fontsize=12)  
        plt.yticks(fontsize=12)

        # Save the plot as a PNG file and display it
        plt.savefig(f'ROC_curve_plot_{self.plot_title}.png', dpi=300, bbox_inches="tight")
       
        # Close the figure to free up resources
        plt.show()
        plt.close()
 
        return

    def compute_rates(self):
        # Initialize lists for False Positive Rate (FPR), False Negative Rate (FNR), and True Positive Rate (TPR)
        # Iterate through threshold values and calculate TP, FP, TN, and FN for each threshold
        # Calculate FPR, FNR, and TPR based on the obtained values
        # Append calculated rates to their respective lists
        # Return the lists of FPR, FNR, and TPR
        FPR, FNR, TPR = [], [], []
        for threshold in self.thresholds:
            TP = np.sum(self.genuine_scores >= threshold)
            FP = np.sum(self.impostor_scores >= threshold)
            TN = np.sum(self.impostor_scores < threshold)
            FN = np.sum(self.genuine_scores < threshold)
            FPR.append(FP / (FP + TN + self.epsilon))
            FNR.append(FN / (FN + TP + self.epsilon))
            TPR.append(TP / (TP + FN + self.epsilon))
        return np.array(FPR), np.array(FNR), np.array(TPR)