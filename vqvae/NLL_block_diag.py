import torch
from torch.distributions import MultivariateNormal

def mvg_nll_block(cov,x,mu,n):
    d = x.shape[0]
    num_blocks = d//n
    remainder = d%n

    B = cov[:n,:n]
    L = torch.linalg.cholesky(B)
    mvn = MultivariateNormal((torch.zeros(n, device=device)), scale_tril=L)

    z = x-mu
    log_likelihood = 0.0
    for i in range(num_blocks):
        z_i = z[i * n:(i + 1) * n]  # Extract corresponding data slice
        log_likelihood += mvn.log_prob(z_i)

    if remainder > 0:
        B_rem = cov[:remainder, :remainder].to(device)  # Same Assumption
        L_rem = torch.linalg.cholesky(B_rem)
        mvn_rem = MultivariateNormal(torch.zeros(remainder, device=device), scale_tril=L_rem)
        z_rem = z[num_blocks * n:]
        log_likelihood += mvn_rem.log_prob(z_rem)

    return -log_likelihood