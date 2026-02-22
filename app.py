import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from hmm import HiddenMarkovModel

st.set_page_config(page_title="HMM - Baum Welch", layout="wide")
st.title("Hidden Markov Model - Baum Welch")

# ── Inputs ────────────────────────────────────────────────────────────────────
N = st.number_input("Number of Hidden States", 2, 10, 2)
M = st.number_input("Number of Observation Symbols", 2, 10, 2)
obs_input = st.text_input("Enter observation sequence (comma separated)", "0,1,2,0,0,1,2,1,0,2")
max_iter  = st.slider("Max Iterations", 10, 500, 100)
tol       = st.select_slider("Tolerance", options=[1e-3, 1e-4, 1e-5, 1e-6], value=1e-6)

if st.button("Train Model"):
    try:
        O = list(map(int, obs_input.split(",")))
        T = len(O)
        N = int(N); M = int(max(O) + 1)

        hmm = HiddenMarkovModel(N, M)

        # ── Run Baum-Welch manually to capture history ────────────────────────
        log_likelihoods = []
        A_history, B_history, pi_history = [], [], []
        converged = False

        for iteration in range(max_iter):
            alpha = hmm.forward(O)
            beta  = hmm.backward(O)

            ll = np.log(np.sum(alpha[-1]) + 1e-300)
            log_likelihoods.append(ll)

            gamma = alpha * beta
            gs = gamma.sum(axis=1, keepdims=True)
            gs[gs == 0] = 1e-300
            gamma /= gs

            xi = np.zeros((T - 1, N, N))
            for t in range(T - 1):
                denom = np.sum(alpha[t][:, None] * hmm.A * hmm.B[:, O[t+1]] * beta[t+1])
                for i in range(N):
                    numer = alpha[t, i] * hmm.A[i] * hmm.B[:, O[t+1]] * beta[t+1]
                    xi[t, i] = numer / (denom + 1e-300)

            hmm.pi = gamma[0]
            dA = gamma[:-1].sum(axis=0); dA[dA == 0] = 1e-300
            hmm.A = xi.sum(axis=0) / dA[:, None]
            for k in range(M):
                mask = (np.array(O) == k)
                hmm.B[:, k] = gamma[mask].sum(axis=0)
            dB = gamma.sum(axis=0); dB[dB == 0] = 1e-300
            hmm.B /= dB[:, None]

            A_history.append(hmm.A.copy())
            B_history.append(hmm.B.copy())
            pi_history.append(hmm.pi.copy())

            if iteration > 0 and abs(log_likelihoods[-1] - log_likelihoods[-2]) < tol:
                converged = True
                break

        n_iter   = len(log_likelihoods)
        final_ll = log_likelihoods[-1]
        delta    = abs(log_likelihoods[-1] - log_likelihoods[-2]) if n_iter > 1 else 0
        iters    = list(range(1, n_iter + 1))
        state_labels = [f"S{i}" for i in range(N)]
        obs_labels   = [f"O{k}" for k in range(M)]

        # Final intermediate variables
        fa = hmm.forward(O)
        fb = hmm.backward(O)
        fg = fa * fb
        fgs = fg.sum(axis=1, keepdims=True); fgs[fgs == 0] = 1e-300
        fg /= fgs

        # ── Live Metrics ──────────────────────────────────────────────────────
        st.subheader("Live Metrics")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Iterations", n_iter)
        m2.metric("Log-Likelihood", f"{final_ll:.4f}")
        m3.metric("Δ Change", f"{delta:.3e}")
        m4.metric("Status", "Converged ✅" if converged else "Max Iter ⚠️")

        st.divider()

        # ── Charts ────────────────────────────────────────────────────────────
        st.subheader("Log-Likelihood Convergence — log P(O|λ)")
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(iters, log_likelihoods, color="#4fc3f7", linewidth=2)
        ax.set_xlabel("Iteration"); ax.set_ylabel("log P(O|λ)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig); plt.close()

        st.subheader("Observation Probability — P(O|λ)")
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(iters, [np.exp(l) for l in log_likelihoods], color="#66bb6a", linewidth=2)
        ax.set_xlabel("Iteration"); ax.set_ylabel("P(O|λ)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig); plt.close()

        st.subheader("Optimization Loss — Negative Log-Likelihood")
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(iters, [-l for l in log_likelihoods], color="#ef5350", linewidth=2)
        ax.set_xlabel("Iteration"); ax.set_ylabel("NLL (−log P)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig); plt.close()

        st.divider()

        # ── State Transition Diagram ──────────────────────────────────────────
        st.subheader("HMM State Transition Diagram")

        palette = ["#e67e22", "#2980b9", "#27ae60", "#8e44ad",
                   "#e74c3c", "#16a085", "#d35400", "#7f8c8d"]

        fig, ax = plt.subplots(figsize=(11, 6))
        ax.set_xlim(0, 11); ax.set_ylim(0, 6.5); ax.axis("off")
        ax.set_facecolor("#f9f9f9")

        # State positions arranged in a circle
        positions = {}
        for i in range(N):
            angle = np.pi / 2 + 2 * np.pi * i / N
            cx = 4.5 + 2.2 * np.cos(angle)
            cy = 3.8 + 1.8 * np.sin(angle)
            positions[i] = (cx, cy)
            circle = plt.Circle((cx, cy), 0.55, color=palette[i % len(palette)], zorder=3)
            ax.add_patch(circle)
            ax.text(cx, cy, f"S{i}", ha='center', va='center',
                    fontsize=13, fontweight='bold', color='white', zorder=4)

        # Transition arrows
        for i in range(N):
            xi_, yi = positions[i]
            for j in range(N):
                xj, yj = positions[j]
                prob = hmm.A[i, j]
                if prob < 0.01:
                    continue
                if i == j:
                    loop = mpatches.Arc(
                        (xi_ + 0.45, yi + 0.45), 0.65, 0.65,
                        angle=0, theta1=0, theta2=310,
                        color=palette[i % len(palette)], linewidth=1.5, zorder=2
                    )
                    ax.add_patch(loop)
                    ax.text(xi_ + 0.9, yi + 0.85, f"{prob:.2f}",
                            fontsize=8, ha='center', color=palette[i % len(palette)])
                else:
                    dx, dy = xj - xi_, yj - yi
                    length = np.sqrt(dx**2 + dy**2)
                    ux, uy = dx / length, dy / length
                    sx = xi_ + ux * 0.56 + uy * 0.12
                    sy = yi  + uy * 0.56 - ux * 0.12
                    ex = xj  - ux * 0.56 + uy * 0.12
                    ey = yj  - uy * 0.56 - ux * 0.12
                    ax.annotate("", xy=(ex, ey), xytext=(sx, sy),
                                arrowprops=dict(arrowstyle="-|>", color="#555",
                                                lw=max(0.5, 2.0 * prob)))
                    mx = (sx + ex) / 2 + uy * 0.25
                    my = (sy + ey) / 2 - ux * 0.25
                    ax.text(mx, my, f"{prob:.2f}", fontsize=8,
                            ha='center', color="#333")

        # Observation nodes
        obs_y = 0.9
        obs_xs = np.linspace(1.2, 9.8, M)
        for k in range(M):
            rect = plt.Rectangle((obs_xs[k] - 0.38, obs_y - 0.25), 0.76, 0.5,
                                  color='#dce3f0', ec='#aaa', zorder=3, linewidth=1.2)
            ax.add_patch(rect)
            ax.text(obs_xs[k], obs_y, f"O{k}", ha='center', va='center',
                    fontsize=10, color='#222', zorder=4)
            for i in range(N):
                xi_, yi = positions[i]
                prob = hmm.B[i, k]
                if prob < 0.05:
                    continue
                ax.annotate("", xy=(obs_xs[k], obs_y + 0.25),
                            xytext=(xi_, yi - 0.56),
                            arrowprops=dict(arrowstyle="-|>",
                                            color=palette[i % len(palette)],
                                            lw=0.8, alpha=0.6,
                                            connectionstyle="arc3,rad=0.15"))
                mx = (xi_ + obs_xs[k]) / 2
                my = (yi - 0.56 + obs_y + 0.25) / 2
                ax.text(mx, my, f"{prob:.2f}",
                        fontsize=7, color=palette[i % len(palette)], alpha=0.9)

        ax.text(0.2, 3.8, "HIDDEN\nSTATES", fontsize=7, color='#888',
                va='center', ha='left')
        ax.text(0.2, 0.9, "OBSERVATIONS", fontsize=7, color='#888',
                va='center', ha='left')
        ax.axhline(y=1.7, color='#ccc', linewidth=0.8, linestyle='--', xmin=0.02, xmax=0.98)

        fig.tight_layout()
        st.pyplot(fig); plt.close()

        st.divider()

        # ── Heatmaps ──────────────────────────────────────────────────────────
        st.subheader("Transition Matrix A & Emission Matrix B")
        hc1, hc2 = st.columns(2)

        with hc1:
            fig, ax = plt.subplots(figsize=(4, 3))
            im = ax.imshow(hmm.A, cmap='Blues', vmin=0, vmax=1)
            ax.set_xticks(range(N)); ax.set_xticklabels(state_labels)
            ax.set_yticks(range(N)); ax.set_yticklabels(state_labels)
            for i in range(N):
                for j in range(N):
                    ax.text(j, i, f"{hmm.A[i,j]:.3f}", ha='center', va='center',
                            color='white' if hmm.A[i,j] > 0.5 else 'black', fontsize=9)
            ax.set_title("Transition Matrix A")
            plt.colorbar(im, ax=ax)
            fig.tight_layout()
            st.pyplot(fig); plt.close()

        with hc2:
            fig, ax = plt.subplots(figsize=(max(4, M), 3))
            im = ax.imshow(hmm.B, cmap='Blues', vmin=0, vmax=1)
            ax.set_xticks(range(M)); ax.set_xticklabels(obs_labels)
            ax.set_yticks(range(N)); ax.set_yticklabels(state_labels)
            for i in range(N):
                for k in range(M):
                    ax.text(k, i, f"{hmm.B[i,k]:.3f}", ha='center', va='center',
                            color='white' if hmm.B[i,k] > 0.5 else 'black', fontsize=9)
            ax.set_title("Emission Matrix B")
            plt.colorbar(im, ax=ax)
            fig.tight_layout()
            st.pyplot(fig); plt.close()

        st.divider()

        # ── Initial Probabilities ─────────────────────────────────────────────
        st.subheader("Initial Probabilities (π)")
        fig, ax = plt.subplots(figsize=(8, max(2, N * 0.5)))
        bars = ax.barh(state_labels, hmm.pi, color="#4fc3f7")
        ax.set_xlim(0, 1.15)
        for bar, v in zip(bars, hmm.pi):
            ax.text(v + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{v:.4f}", va='center', fontsize=9)
        ax.set_xlabel("Probability")
        ax.grid(True, axis='x', alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig); plt.close()

        st.divider()

        # ── Parameter Evolution ───────────────────────────────────────────────
        st.subheader("Parameter Evolution — A[i][j] over Iterations")
        fig, ax = plt.subplots(figsize=(10, 3.5))
        colors_ev = ["#4fc3f7", "#ef5350", "#66bb6a", "#ce93d8",
                     "#ffb74d", "#80cbc4", "#f48fb1", "#bcaaa4"]
        c = 0
        for i in range(N):
            for j in range(N):
                vals = [A_history[it][i, j] for it in range(len(A_history))]
                ax.plot(iters, vals, label=f"A[{i}][{j}]",
                        color=colors_ev[c % len(colors_ev)], linewidth=1.5)
                c += 1
        ax.set_xlabel("Iteration"); ax.set_ylabel("Probability")
        ax.legend(fontsize=8, loc='right')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig); plt.close()

        st.divider()

        # ── Iteration Log ─────────────────────────────────────────────────────
        st.subheader("Iteration Log")
        rows = []
        for it in range(n_iter):
            row = {
                "Iter": it + 1,
                "Log-Likelihood": round(log_likelihoods[it], 6),
                "ΔLL": round(abs(log_likelihoods[it] - log_likelihoods[it-1]), 6) if it > 0 else None
            }
            for i in range(N):
                for j in range(N):
                    row[f"A[{i}][{j}]"] = round(A_history[it][i, j], 4)
            for i in range(N):
                for k in range(M):
                    row[f"B[{i}][{k}]"] = round(B_history[it][i, k], 4)
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), use_container_width=True, height=260)

        st.divider()

        # ── Intermediate Variables ────────────────────────────────────────────
        st.subheader("Intermediate Variables (Final Iteration)")
        disp = min(20, T)

        def make_df(arr):
            df = pd.DataFrame(arr[:disp], columns=state_labels)
            df.insert(0, "t", range(disp))
            return df

        tab_a, tab_b, tab_g = st.tabs(["Alpha (α)", "Beta (β)", "Gamma (γ)"])
        with tab_a:
            st.caption("Showing first 20 time steps.")
            st.dataframe(make_df(fa), use_container_width=True)
        with tab_b:
            st.caption("Showing first 20 time steps.")
            st.dataframe(make_df(fb), use_container_width=True)
        with tab_g:
            st.caption("Showing first 20 time steps.")
            st.dataframe(make_df(fg), use_container_width=True)

        st.divider()

        # ── Final Learned Parameters ──────────────────────────────────────────
        st.subheader("Final Learned Parameters")
        fp1, fp2, fp3, fp4 = st.columns(4)
        fp1.metric("Total Iterations", n_iter)
        fp2.metric("Final log P(O|λ)", f"{final_ll:.6f}")
        fp3.metric("Final P(O|λ)", f"{np.exp(final_ll):.4e}")
        fp4.metric("Converged", "Yes ✅" if converged else "No ⚠️")

        st.markdown("**Transition Matrix A**")
        st.dataframe(pd.DataFrame(hmm.A, index=state_labels, columns=state_labels).round(6))

        st.markdown("**Emission Matrix B**")
        st.dataframe(pd.DataFrame(hmm.B, index=state_labels, columns=obs_labels).round(6))

        st.markdown("**Initial Distribution π**")
        st.dataframe(pd.DataFrame([hmm.pi], columns=[f"π(S{i})" for i in range(N)]).round(6))

        st.divider()

        # ── HMM Theory Reference ──────────────────────────────────────────────
        st.subheader("HMM Theory & Algorithm Reference")

        with st.expander("1. Hidden Markov Model — Formal Definition", expanded=True):
            st.markdown("""
An HMM is defined by **λ = (A, B, π)** where:

| Symbol | Name | Description |
|--------|------|-------------|
| **A** | Transition matrix | A[i][j] = P(Sⱼ \| Sᵢ) |
| **B** | Emission matrix | B[i][k] = P(Oₖ \| Sᵢ) |
| **π** | Initial distribution | π[i] = P(q₁ = Sᵢ) |
""")

        with st.expander("2. Forward Algorithm (α)"):
            st.markdown("""
**α_t(i) = P(O₁…Oₜ, qₜ=Sᵢ | λ)**

- **Init:** α₁(i) = πᵢ · B[i][O₁]
- **Recursion:** α_{t+1}(j) = [Σᵢ α_t(i) · A[i][j]] · B[j][O_{t+1}]
- **Termination:** P(O|λ) = Σᵢ α_T(i)
""")

        with st.expander("3. Backward Algorithm (β)"):
            st.markdown("""
**β_t(i) = P(O_{t+1}…O_T | qₜ=Sᵢ, λ)**

- **Init:** β_T(i) = 1
- **Recursion:** β_t(i) = Σⱼ A[i][j] · B[j][O_{t+1}] · β_{t+1}(j)
""")

        with st.expander("4. Baum-Welch Re-estimation (EM)"):
            st.markdown("""
- **γ_t(i)** = α_t(i)·β_t(i) / Σⱼ α_t(j)·β_t(j)
- **ξ_t(i,j)** = α_t(i)·A[i][j]·B[j][O_{t+1}]·β_{t+1}(j) / P(O|λ)
- **π̄ᵢ** = γ₁(i)
- **Āᵢⱼ** = Σₜ ξₜ(i,j) / Σₜ γₜ(i)
- **B̄ᵢₖ** = Σ_{t:Oₜ=vₖ} γₜ(i) / Σₜ γₜ(i)

Repeat until **|Δ log P| < tolerance**.
""")

    except Exception as e:
        st.error(f"Error: {e}")
        st.exception(e)