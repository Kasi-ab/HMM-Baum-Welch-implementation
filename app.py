import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from hmm import HiddenMarkovModel

st.set_page_config(page_title="HMM Baum-Welch", layout="wide")

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("## 🔍 HMM Parameter Estimation")
st.caption("Baum-Welch (EM) Algorithm — Forward · Backward · Re-estimation")
st.divider()

# ── Sidebar Inputs ────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    N         = st.number_input("Hidden States (N)", 2, 10, 2)
    obs_input = st.text_area("Observation Sequence", "0,1,2,0,0,1,2,1,0,2",
                              help="Comma-separated integers, 0-based")
    max_iter  = st.slider("Max Iterations", 10, 500, 100)
   tol_exp = st.slider("Convergence Tolerance (10^x)", min_value=-10, max_value=-2, value=-6)
tol = 10 ** tol_exp
st.caption(f"Tolerance: 1e{tol_exp} = {tol}")
    theme     = st.toggle("Dark Charts", value=True)
    st.divider()
    run = st.button("▶ Run Training", use_container_width=True, type="primary")

# ── Theme ─────────────────────────────────────────────────────────────────────
if theme:
    bg, fg, grid, tick = "#0e1117", "#dddddd", "#2a2d3a", "#aaaaaa"
else:
    bg, fg, grid, tick = "#ffffff", "#111111", "#e0e0e0", "#555555"

COLORS = ["#f72585", "#4cc9f0", "#7bed9f", "#ffd166",
          "#a29bfe", "#fd79a8", "#55efc4", "#fdcb6e"]

def styled_fig(figsize=(10, 3)):
    fig, ax = plt.subplots(figsize=figsize, facecolor=bg)
    ax.set_facecolor(bg)
    ax.tick_params(colors=tick, labelsize=8)
    ax.xaxis.label.set_color(fg)
    ax.yaxis.label.set_color(fg)
    for sp in ax.spines.values():
        sp.set_edgecolor(grid)
    return fig, ax

# ── Main ──────────────────────────────────────────────────────────────────────
if run:
    try:
        O = list(map(int, obs_input.split(",")))
        T = len(O)
        N = int(N)
        M = max(O) + 1

        hmm = HiddenMarkovModel(N, M)
        log_likelihoods = []
        converged = False

        for iteration in range(int(max_iter)):
            alpha = hmm.forward(O)
            beta  = hmm.backward(O)
            ll    = np.log(np.sum(alpha[-1]) + 1e-300)
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

            if iteration > 0 and abs(log_likelihoods[-1] - log_likelihoods[-2]) < tol:
                converged = True
                break

        n_iter   = len(log_likelihoods)
        final_ll = log_likelihoods[-1]
        delta    = abs(log_likelihoods[-1] - log_likelihoods[-2]) if n_iter > 1 else 0
        iters    = list(range(1, n_iter + 1))
        state_labels = [f"S{i}" for i in range(N)]
        obs_labels   = [f"O{k}" for k in range(M)]

        fa = hmm.forward(O)
        fb = hmm.backward(O)
        fg_arr = fa * fb
        fgs = fg_arr.sum(axis=1, keepdims=True); fgs[fgs == 0] = 1e-300
        fg_arr /= fgs

        # ── Status Banner ─────────────────────────────────────────────────────
        if converged:
            st.success(f"✅ Converged in **{n_iter}** iterations — "
                       f"log P(O|λ) = **{final_ll:.4f}** — Δ = **{delta:.2e}**")
        else:
            st.warning(f"⚠️ Reached max iterations ({n_iter}) — "
                       f"log P(O|λ) = **{final_ll:.4f}** — Δ = **{delta:.2e}**")

        # ── 3 Charts side by side ─────────────────────────────────────────────
        st.markdown("### 📈 Training Curves")
        ch1, ch2, ch3 = st.columns(3)

        with ch1:
            st.caption("Log-Likelihood")
            fig, ax = styled_fig((5, 2.8))
            ax.plot(iters, log_likelihoods, color=COLORS[0], linewidth=2)
            ax.fill_between(iters, log_likelihoods,
                            min(log_likelihoods), alpha=0.15, color=COLORS[0])
            ax.set_xlabel("Iteration"); ax.set_ylabel("log P(O|λ)")
            ax.grid(True, color=grid, linewidth=0.4)
            fig.tight_layout(); st.pyplot(fig); plt.close()

        with ch2:
            st.caption("Observation Probability P(O|λ)")
            fig, ax = styled_fig((5, 2.8))
            probs = [np.exp(l) for l in log_likelihoods]
            ax.plot(iters, probs, color=COLORS[1], linewidth=2)
            ax.fill_between(iters, probs, 0, alpha=0.15, color=COLORS[1])
            ax.set_xlabel("Iteration"); ax.set_ylabel("P(O|λ)")
            ax.grid(True, color=grid, linewidth=0.4)
            fig.tight_layout(); st.pyplot(fig); plt.close()

        with ch3:
            st.caption("Negative Log-Likelihood (Loss)")
            fig, ax = styled_fig((5, 2.8))
            nll = [-l for l in log_likelihoods]
            ax.plot(iters, nll, color=COLORS[3], linewidth=2)
            ax.fill_between(iters, nll, min(nll), alpha=0.15, color=COLORS[3])
            ax.set_xlabel("Iteration"); ax.set_ylabel("−log P(O|λ)")
            ax.grid(True, color=grid, linewidth=0.4)
            fig.tight_layout(); st.pyplot(fig); plt.close()

        st.divider()

        # ── State Transition Diagram ──────────────────────────────────────────
        st.markdown("### 🔗 State Transition Diagram")

        obs_box_bg = "#1e2130" if theme else "#eef2ff"
        obs_box_ec = "#444"    if theme else "#aaa"
        obs_txt    = "white"   if theme else "#222"
        lbl_col    = "#cccccc" if theme else "#333333"
        arr_col    = "#888888" if theme else "#555555"

        fig, ax = plt.subplots(figsize=(11, 6), facecolor=bg)
        ax.set_facecolor(bg)
        ax.set_xlim(0, 11); ax.set_ylim(0, 6.5); ax.axis("off")

        positions = {}
        for i in range(N):
            angle = np.pi / 2 + 2 * np.pi * i / N
            cx = 4.8 + 2.4 * np.cos(angle)
            cy = 3.8 + 1.8 * np.sin(angle)
            positions[i] = (cx, cy)
            # Outer glow ring
            ring = plt.Circle((cx, cy), 0.65, color=COLORS[i % len(COLORS)],
                               alpha=0.2, zorder=2)
            ax.add_patch(ring)
            # Main node
            circle = plt.Circle((cx, cy), 0.52, color=COLORS[i % len(COLORS)], zorder=3)
            ax.add_patch(circle)
            ax.text(cx, cy, f"S{i}", ha='center', va='center',
                    fontsize=12, fontweight='bold', color='white', zorder=4)

        for i in range(N):
            xi_, yi = positions[i]
            for j in range(N):
                xj, yj = positions[j]
                prob = hmm.A[i, j]
                if prob < 0.01:
                    continue
                if i == j:
                    loop = mpatches.Arc(
                        (xi_ + 0.42, yi + 0.42), 0.6, 0.6,
                        angle=0, theta1=0, theta2=310,
                        color=COLORS[i % len(COLORS)], linewidth=1.8, zorder=2
                    )
                    ax.add_patch(loop)
                    ax.text(xi_ + 0.85, yi + 0.82, f"{prob:.2f}",
                            fontsize=8, ha='center', color=COLORS[i % len(COLORS)])
                else:
                    dx, dy = xj - xi_, yj - yi
                    ln = np.sqrt(dx**2 + dy**2)
                    ux, uy = dx/ln, dy/ln
                    sx = xi_ + ux*0.53 + uy*0.12
                    sy = yi  + uy*0.53 - ux*0.12
                    ex = xj  - ux*0.53 + uy*0.12
                    ey = yj  - uy*0.53 - ux*0.12
                    ax.annotate("", xy=(ex, ey), xytext=(sx, sy),
                                arrowprops=dict(arrowstyle="-|>", color=arr_col,
                                                lw=max(0.6, 2.2 * prob)))
                    mx = (sx+ex)/2 + uy*0.28
                    my = (sy+ey)/2 - ux*0.28
                    ax.text(mx, my, f"{prob:.2f}", fontsize=8,
                            ha='center', color=lbl_col)

        obs_y  = 0.85
        obs_xs = np.linspace(1.2, 9.8, M)
        for k in range(M):
            rect = plt.Rectangle((obs_xs[k]-0.4, obs_y-0.27), 0.8, 0.54,
                                  color=obs_box_bg, ec=obs_box_ec,
                                  zorder=3, linewidth=1.2, linestyle='--')
            ax.add_patch(rect)
            ax.text(obs_xs[k], obs_y, f"O{k}", ha='center', va='center',
                    fontsize=10, color=obs_txt, zorder=4)
            for i in range(N):
                xi_, yi = positions[i]
                prob = hmm.B[i, k]
                if prob < 0.05:
                    continue
                ax.annotate("", xy=(obs_xs[k], obs_y+0.27),
                            xytext=(xi_, yi-0.53),
                            arrowprops=dict(arrowstyle="-|>",
                                            color=COLORS[i % len(COLORS)],
                                            lw=0.9, alpha=0.55,
                                            connectionstyle="arc3,rad=0.2"))
                ax.text((xi_+obs_xs[k])/2, (yi-0.53+obs_y+0.27)/2,
                        f"{prob:.2f}", fontsize=7,
                        color=COLORS[i % len(COLORS)], alpha=0.9)

        ax.axhline(y=1.75, color=grid, linewidth=0.8,
                   linestyle=':', xmin=0.02, xmax=0.98)
        ax.text(0.15, 3.8, "HIDDEN\nSTATES", fontsize=7,
                color=lbl_col, va='center', alpha=0.7)
        ax.text(0.15, 0.85, "EMITTED\nSYMBOLS", fontsize=7,
                color=lbl_col, va='center', alpha=0.7)

        fig.tight_layout()
        st.pyplot(fig); plt.close()

        st.divider()

        # ── Intermediate Variables ────────────────────────────────────────────
        st.markdown("### 🔬 Intermediate Variables (Final Iteration)")
        disp = min(20, T)

        def make_df(arr):
            df = pd.DataFrame(arr[:disp], columns=state_labels)
            df.insert(0, "t", range(disp))
            return df

        tab_a, tab_b, tab_g = st.tabs(["α  Forward", "β  Backward", "γ  Gamma"])
        with tab_a:
            st.caption("α_t(i) = P(O₁…Oₜ, qₜ=Sᵢ | λ)  — first 20 steps")
            st.dataframe(make_df(fa), use_container_width=True)
        with tab_b:
            st.caption("β_t(i) = P(O_{t+1}…O_T | qₜ=Sᵢ, λ)  — first 20 steps")
            st.dataframe(make_df(fb), use_container_width=True)
        with tab_g:
            st.caption("γ_t(i) = P(qₜ=Sᵢ | O, λ)  — first 20 steps")
            st.dataframe(make_df(fg_arr), use_container_width=True)

        st.divider()

        # ── Final Parameters ──────────────────────────────────────────────────
        st.markdown("### 📋 Learned Parameters")
        pc1, pc2 = st.columns(2)
        with pc1:
            st.markdown("**Transition Matrix A**")
            st.dataframe(pd.DataFrame(hmm.A, index=state_labels,
                                      columns=state_labels).round(6),
                         use_container_width=True)
        with pc2:
            st.markdown("**Emission Matrix B**")
            st.dataframe(pd.DataFrame(hmm.B, index=state_labels,
                                      columns=obs_labels).round(6),
                         use_container_width=True)

        st.markdown("**Initial Distribution π**")
        st.dataframe(pd.DataFrame([hmm.pi],
                     columns=[f"π(S{i})" for i in range(N)]).round(6),
                     use_container_width=True)

        st.divider()

        # ── Theory ───────────────────────────────────────────────────────────
        st.markdown("### 📖 Algorithm Reference")

        with st.expander("1. Formal Definition  λ = (A, B, π)"):
            st.markdown("""
| Symbol | Name | Description |
|--------|------|-------------|
| **A** | Transition matrix | A[i][j] = P(Sⱼ \| Sᵢ) |
| **B** | Emission matrix | B[i][k] = P(Oₖ \| Sᵢ) |
| **π** | Initial distribution | π[i] = P(q₁ = Sᵢ) |
""")
        with st.expander("2. Forward Algorithm  α"):
            st.markdown("""
- **Init:** α₁(i) = πᵢ · B[i][O₁]
- **Recursion:** α_{t+1}(j) = [Σᵢ α_t(i) · A[i][j]] · B[j][O_{t+1}]
- **Result:** P(O|λ) = Σᵢ α_T(i)
""")
        with st.expander("3. Backward Algorithm  β"):
            st.markdown("""
- **Init:** β_T(i) = 1
- **Recursion:** β_t(i) = Σⱼ A[i][j] · B[j][O_{t+1}] · β_{t+1}(j)
""")
        with st.expander("4. Baum-Welch Re-estimation"):
            st.markdown("""
- **γ_t(i)** = α_t(i)·β_t(i) / Σⱼ α_t(j)·β_t(j)
- **π̄ᵢ** = γ₁(i)
- **Āᵢⱼ** = Σₜ ξₜ(i,j) / Σₜ γₜ(i)
- **B̄ᵢₖ** = Σ_{t:Oₜ=k} γₜ(i) / Σₜ γₜ(i)

Repeat until **|Δ log P| < tolerance**.
""")

    except Exception as e:
        st.error(f"Error: {e}")
        st.exception(e)

else:
    st.info("👈 Configure your model in the sidebar and click **▶ Run Training** to start.")
