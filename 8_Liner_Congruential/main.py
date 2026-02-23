#1.The LCG(linear congruential generators) is an algorithm for generating random looking numbers
###It is used in quant for random looking numbers for simulating asset paths

### Xn+1 = (aXn + c) mod m

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def lcg(seed, a, c, m, size):
    sequence = np.zeros(size) #initialize the sequence array with zeros
    sequence[0] = seed        # initial value

    for i in range(1, size):
        sequence[i] = (a * sequence[i-1] + c) % m
    
    return sequence


if __name__ == "__main__":
    ##################### Example ##############
    seed = 42
    a    = 1664525
    c    = 1013904223
    m    = 2**32
    size = 1000

    random_sequence = lcg(seed, a, c, m, size)

    ################## Graphs ###############
    plt.hist(random_sequence, bins=50, edgecolor='black', alpha=0.7)
    plt.title("Lcg Numbers")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.show()


    plt.scatter(random_sequence[:-1], random_sequence[1:], alpha=0.5)
    plt.title("Scatter Plot of Successive LCG Values")
    plt.xlabel("X_n")
    plt.ylabel("X_n+1")
    plt.show()

    # Convert the sequence to a Pandas Series
    series = pd.Series(random_sequence)
    pd.plotting.autocorrelation_plot(series)
    plt.title("Autocorrelation of LCG Generated Sequence")
    plt.show()