import matplotlib.pyplot as plt
import seaborn as sns


def plot_steps_and_rewards(df):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    # ax[0].set(ylabel="Cummulated rewards")
    sns.lineplot(data=df, x="Episodes", y="Rewards", ax=ax[0])
    # ax[0].set(ylabel="Cumulated rewards")

    sns.lineplot(data=df, x="Episodes", y="Steps", ax=ax[1])
    # ax[1].set(ylabel="Averaged steps number")

    plt.show()
