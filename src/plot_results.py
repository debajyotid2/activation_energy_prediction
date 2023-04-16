import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

RESULTS_FILE = "results_no_rev_in_test.csv"

def main():
    results = pd.read_csv(RESULTS_FILE, index_col=0)
    colnames = ["MAE train", "MAE test", "MSE train",
                "MSE test", "R^2 train", "R^2 test"]
    for col in colnames:
        sns.regplot(results, x="Dataset size", y=col)
        plt.show()

if __name__ == "__main__":
    main()
