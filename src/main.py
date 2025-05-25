import data_preparation
import feature_extraction
import clustering
import classification
import visualization

import warnings
# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

def main():
    # Prepare the data 
    data_preparation.main()
    # Extract features
    feature_extraction.main()
    # Perform clustering
    clustering.main()
    # Perform classification
    classification.main()
    # Visualize results
    visualization.main()

if __name__ == '__main__':
    main()