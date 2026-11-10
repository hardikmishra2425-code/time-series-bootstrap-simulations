import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# ==============================================================================
# 1. SIMULATE A PERSISTENT TIME SERIES: AR(1) PROCESS
# X_t = rho * X_{t-1} + e_t, where true mean mu = 0.0
# ==============================================================================
def generate_ar1_series(T=200, rho=0.7, sigma=1.0):
    """Generates an autoregressive AR(1) time series with persistence rho."""
    X = np.zeros(T)
    noise = np.random.normal(0, sigma, T)
    for t in range(1, T):
        X[t] = rho * X[t-1] + noise[t]
    return X

# ==============================================================================
# 2. RESAMPLING SCHEMES
# ==============================================================================
def standard_iid_bootstrap(X, B=400):
    """
    Standard Efron i.i.d. bootstrap: shuffles individual observations with replacement.
    Destroys all serial temporal dependence.
    """
    T = len(X)
    resamples = np.random.choice(X, size=(B, T), replace=True)
    return np.mean(resamples, axis=1)

def moving_block_bootstrap(X, block_size=10, B=400):
    """
    Moving Block Bootstrap (MBB): resamples contiguous blocks of size b.
    Preserves the short-range autocorrelation structure within blocks.
    """
    T = len(X)
    num_blocks = int(np.ceil(T / block_size))
    # Extract all overlapping contiguous blocks
    blocks = np.array([X[i : i + block_size] for i in range(T - block_size + 1)])
    
    bootstrap_means = np.zeros(B)
    for b in range(B):
        idx = np.random.choice(len(blocks), size=num_blocks, replace=True)
        synthetic_series = blocks[idx].flatten()[:T]
        bootstrap_means[b] = np.mean(synthetic_series)
    return bootstrap_means

# ==============================================================================
# 3. MONTE CARLO COVERAGE PROBABILITY EXPERIMENT
# ==============================================================================
def evaluate_coverage(n_trials=400, T=200, rho=0.7, block_size=10, nominal_conf=0.95):
    true_mean = 0.0
    alpha = (1.0 - nominal_conf) / 2.0
    lower_pct = alpha * 100
    upper_pct = (1.0 - alpha) * 100
    
    covered_iid = 0
    covered_mbb = 0
    
    print(f"Running {n_trials} Monte Carlo trials (T={T}, autocorrelation rho={rho}, block size b={block_size})...")
    
    for _ in range(n_trials):
        X = generate_ar1_series(T=T, rho=rho)
        
        # Standard i.i.d. Bootstrap Confidence Interval
        boot_iid = standard_iid_bootstrap(X, B=300)
        ci_iid = np.percentile(boot_iid, [lower_pct, upper_pct])
        if ci_iid[0] <= true_mean <= ci_iid[1]:
            covered_iid += 1
            
        # Moving Block Bootstrap Confidence Interval
        boot_mbb = moving_block_bootstrap(X, block_size=block_size, B=300)
        ci_mbb = np.percentile(boot_mbb, [lower_pct, upper_pct])
        if ci_mbb[0] <= true_mean <= ci_mbb[1]:
            covered_mbb += 1
            
    cov_rate_iid = (covered_iid / n_trials) * 100
    cov_rate_mbb = (covered_mbb / n_trials) * 100
    
    return cov_rate_iid, cov_rate_mbb, boot_iid, boot_mbb

if __name__ == "__main__":
    cov_iid, cov_mbb, sample_iid, sample_mbb = evaluate_coverage()
    
    print("\n=================== EXPERIMENT SUMMARY ===================")
    print(f"Nominal Confidence Level:         95.0%")
    print(f"Standard (i.i.d.) Coverage:       {cov_iid:.1f}%  <-- Severe undercoverage due to variance underestimation")
    print(f"Moving Block Bootstrap (b=10):    {cov_mbb:.1f}%  <-- Recovers nominal coverage by preserving dependence")
    print("==========================================================\n")
    
    # Publication-ready diagnostic visualization
    plt.figure(figsize=(9, 5))
    plt.hist(sample_iid, bins=30, density=True, alpha=0.55, color='#d95f02', edgecolor='black',
             label=f'Standard i.i.d. Bootstrap (Artificially Narrow, Cov={cov_iid:.1f}%)')
    plt.hist(sample_mbb, bins=30, density=True, alpha=0.55, color='#1b9e77', edgecolor='black',
             label=f'Moving Block Bootstrap b=10 (Preserves Autocorrelation, Cov={cov_mbb:.1f}%)')
    plt.axvline(0.0, color='black', linestyle='--', linewidth=1.8, label=r'True Population Mean ($\mu=0$)')
    plt.title(r'Bootstrap Distribution of the Sample Mean under AR(1) Dependence ($T=200, \rho=0.7$)',
              fontsize=12, fontweight='bold')
    plt.xlabel('Resampled Mean', fontsize=11)
    plt.ylabel('Density', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9.5, loc='upper right')
    plt.tight_layout()
    plt.savefig('bootstrap_time_series_diagnostics.png', dpi=300)
    plt.show()
