## 1.Interest rate derivatives pricing over time

## Vasicek Model is a direct appication of  Ornstein-Uhlenbeck(OU)
### Define- The model belongs to a family of models known as 'mean-reverting' processes, which suggests that interest rates tend to move toward a long-term average level, even though they exhibit short-term fluctuations.



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def vasicek_model(theta, mu, sigma, r0, T, dt):
    N      = int(T / dt) #number of time steps
    rates  = np.zeros(N) #Pre-allocate an array large enough for the sample path
    rates[0] = r0        #set the initial rate

    for t in range(1, N): # skip the initial rate
        dr = theta * (mu - rates[t - 1]) * dt + sigma * np.sqrt(dt) * np.random.normal()
        rates[t] = rates[t-1] + dr

    return rates

def simulate_vasick_paths(theta, mu, sigma, r0, T, dt, num_simulations):
    N = int(T / dt)
    all_simulations = np.zeros((N, num_simulations))

    for i in range(num_simulations):
        all_simulations[:, i] = vasicek_model(theta, mu, sigma, r0, T, dt)

    return pd.DataFrame(all_simulations, columns=[f'Simulation {i+1}' for i in range(num_simulations)])

def plot_vasicek_paths(df, T, dt):
    plt.figure(figsize=(12, 6))
    time_points = np.linspace(0, T, int(T / dt))

    for column in df.columns:
        plt.plot(time_points, df[column], lw=1.0, alpha = 0.6)

    plt.title('Vasicek Interest rate')
    plt.xlabel('Time')
    plt.ylabel('Interest Rate')
    plt.xlim(0.0, 1.0)
    plt.show()


if __name__ == "__main__":
    theta = 2.0
    mu    = 0.05
    sigma = 0.02
    r0    = 0.03
    T     = 1.0
    dt    = 0.001
    num_simulations = 10

    simulated_paths = simulate_vasick_paths(theta, mu, sigma, r0, T, dt, num_simulations)

    plot_vasicek_paths(simulated_paths, T, dt)