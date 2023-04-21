
import textwrap

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

RESULTS_FILES = ["results/forward only.csv", 
                 "results/forward and backward.csv", 
                 "results/scaffold.csv"]

def main():
    plt.rcParams["font.size"] = 14

    colnames = ["Test MAE", "Test MSE", "Test R^2"]
    ylabels = ["Test MAE (kcal/mol)",
               r"Test MSE ($kcal^2/mol^2$)",
               r"Test $R^2$"]

    for count, filename in enumerate(RESULTS_FILES):
        data = pd.read_csv(filename)
        data["experiment"] = filename.split("/")[1].split(".")[0]
        if count == 0:
           all_data = data
        else:
            all_data = pd.concat([all_data, data])
    
    for count, colname in enumerate(colnames):
        fgrid = sns.catplot(data=all_data, x="Regressor", y=colname, 
                    kind="bar", hue="experiment", height=10,
                    aspect=10/10)
        ax = plt.gca()
        ax.set_xticklabels(["\n".join(textwrap.wrap(label, 20))
                      for label in data["Regressor"].unique().tolist()], 
                           rotation=45,
                           ha="right")
        ax.set_ylabel(ylabels[count])
        sns.move_legend(fgrid,
                        ncol=len(all_data["experiment"].unique()),
                        loc="upper center",
                        title=None,
                        bbox_to_anchor=(0.5, 1.00))
        plt.tight_layout()
        plt.savefig(f"{colname}.png")

if __name__ == "__main__":
    main()
