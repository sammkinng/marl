import pandas as pd
import matplotlib.pyplot as plt
import os

# CSV file from previous step
csv_file = './logs_extended/baseline_results.csv'
df = pd.read_csv(csv_file)

output_dir = './logs_extended/plots/'
os.makedirs(output_dir, exist_ok=True)

# Get unique distances
distances = df['distance_km'].unique()

for distance in distances:
    df_dist = df[df['distance_km'] == distance]

    # Group by mu_signal and compute mean/std over seeds
    grouped = df_dist.groupby('mu_signal').agg(
        SKR_mean=('SKR_bits_per_s', 'mean'),
        SKR_std=('SKR_bits_per_s', 'std'),
        QBER_mean=('QBER', 'mean'),
        QBER_std=('QBER', 'std')
    ).reset_index()

    # Plot SKR vs mu_signal
    plt.figure(figsize=(8,5))
    plt.errorbar(grouped['mu_signal'], grouped['SKR_mean'], yerr=grouped['SKR_std'],
                 fmt='o-', capsize=5, label=f'Distance {distance} km')
    plt.xlabel('μ (mu_signal)')
    plt.ylabel('SKR [bits/s]')
    plt.title(f'SKR vs μ (Distance {distance} km)')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'SKR_vs_mu_distance_{distance}.png'))
    plt.close()

    # Plot QBER vs mu_signal
    plt.figure(figsize=(8,5))
    plt.errorbar(grouped['mu_signal'], grouped['QBER_mean'], yerr=grouped['QBER_std'],
                 fmt='s-', capsize=5, color='red', label=f'Distance {distance} km')
    plt.xlabel('μ (mu_signal)')
    plt.ylabel('QBER')
    plt.title(f'QBER vs μ (Distance {distance} km)')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'QBER_vs_mu_distance_{distance}.png'))
    plt.close()

    print(f"Plots saved for distance {distance} km in {output_dir}")
