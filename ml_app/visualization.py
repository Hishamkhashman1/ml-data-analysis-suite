import matplotlib.pyplot as plt
import seaborn as sns


def create_pairplot(df, output_path: str = "pairplot.png") -> str:
    """Create and show a pairplot for a dataframe."""
    sns.pairplot(df)
    plt.savefig(output_path)
    plt.show()
    return output_path
