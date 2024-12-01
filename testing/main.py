from graph import Evaluator
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

def main():
    
    # Set the random seed to 1.
    np.random.seed(1)

    # Name the systems A, B, and C
    systems = ['A', 'B', 'C']

    for system in systems:
        # Use np.random.random sample() to generate a random float between 
        # 0.5 and 0.9 and another random float between 0.0 and 0.2. Use these 
        # as the μ (mean) and σ (standard deviation), respectively, to generate 
        # 400 genuine scores using np.random.normal()
        genuine_mean = np.random.uniform(0.5, 0.9)
        genuine_std = np.random.uniform(0.0, 0.2)
        genuine_scores = np.random.normal(genuine_mean, genuine_std, 400)
        
        # Repeat with μ ∈ [0.1, 0.5) and σ ∈ [0.0, 0.2) to generate 1,600 
        # impostor scores
        impostor_mean = np.random.uniform(0.1, 0.5)
        impostor_std = np.random.uniform(0.0, 0.2)
        impostor_scores = np.random.normal(impostor_mean, impostor_std, 1600)
        
        # Creating an instance of the Evaluator class
        evaluator = Evaluator(
            epsilon=1e-12,
            num_thresholds=200,
            genuine_scores=genuine_scores,
            impostor_scores=impostor_scores,
            plot_title="%s" % system
        )
        
        # Generate the FPR, FNR, and TPR using 200 threshold values equally spaced
        # between -0.1 and 1.1.
        FPR, FNR, TPR = evaluator.compute_rates()
    
        # Plot the score distribution. Include the d-prime value in the plot’s 
        # title. Your genuine scores should be green, and your impostor scores 
        # should be red. Set the x axis limits from -0.05 to 1.05
        evaluator.plot_score_distribution()
                
        # Plot the DET curve and include the EER in the plot’s title. 
        # Set the x and y axes limits from -0.05 to 1.05.
        evaluator.plot_det_curve(FPR, FNR)
        # Plot the ROC curve. Set the x and y axes limits from -0.05 to 1.05.
        evaluator.plot_roc_curve(FPR, TPR)

        
if __name__ == "__main__":
    main()

