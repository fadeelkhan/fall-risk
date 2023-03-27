import matplotlib.pyplot as plt
def plot_falls_and_activities(times, fall_vs_no_fall_predictions):
    plt.plot(times, fall_vs_no_fall_predictions)
    plt.xlabel('Time')
    plt.ylabel('Fall (1) vs. No Fall (0)')
    plt.show()