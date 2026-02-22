import numpy as np

class HiddenMarkovModel:
    def __init__(self, N, M):
        self.N = N
        self.M = M

        self.A = np.random.rand(N, N)
        self.A /= self.A.sum(axis=1, keepdims=True)

        self.B = np.random.rand(N, M)
        self.B /= self.B.sum(axis=1, keepdims=True)

        self.pi = np.random.rand(N)
        self.pi /= self.pi.sum()

    def forward(self, O):
        T = len(O)
        alpha = np.zeros((T, self.N))
        alpha[0] = self.pi * self.B[:, O[0]]

        for t in range(1, T):
            for j in range(self.N):
                alpha[t, j] = np.sum(alpha[t-1] * self.A[:, j]) * self.B[j, O[t]]
        return alpha

    def backward(self, O):
        T = len(O)
        beta = np.zeros((T, self.N))
        beta[T-1] = np.ones(self.N)

        for t in reversed(range(T-1)):
            for i in range(self.N):
                beta[t, i] = np.sum(self.A[i] * self.B[:, O[t+1]] * beta[t+1])
        return beta

    def baum_welch(self, O, max_iter=50):
        T = len(O)
        log_likelihoods = []

        for _ in range(max_iter):
            alpha = self.forward(O)
            beta = self.backward(O)

            # Log-likelihood
            likelihood = np.sum(alpha[-1])
            log_likelihoods.append(np.log(likelihood + 1e-300))  # safer epsilon

            # Gamma — guard against zero rows to prevent NaN
            gamma = alpha * beta
            gamma_sum = gamma.sum(axis=1, keepdims=True)
            gamma_sum[gamma_sum == 0] = 1e-300   # avoid division by zero
            gamma /= gamma_sum

            # Xi
            xi = np.zeros((T-1, self.N, self.N))
            for t in range(T-1):
                denom = np.sum(alpha[t][:, None] * self.A * self.B[:, O[t+1]] * beta[t+1])
                denom = max(denom, 1e-300)        # avoid division by zero
                for i in range(self.N):
                    numer = alpha[t, i] * self.A[i] * self.B[:, O[t+1]] * beta[t+1]
                    xi[t, i] = numer / denom

            # Update π
            self.pi = gamma[0]

            # Update A — guard denominator
            denom_A = gamma[:-1].sum(axis=0)
            denom_A[denom_A == 0] = 1e-300
            self.A = xi.sum(axis=0) / denom_A[:, None]

            # Update B — guard denominator
            for k in range(self.M):
                mask = (np.array(O) == k)
                self.B[:, k] = gamma[mask].sum(axis=0)
            denom_B = gamma.sum(axis=0)
            denom_B[denom_B == 0] = 1e-300
            self.B /= denom_B[:, None]

        return log_likelihoods