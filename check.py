import numpy as np, pickle, matplotlib.pyplot as plt
import cycle_tools as ct

B = np.load("results/cycle_21/field.npy") * 10.0          # native units, (401, 181)
T = 10.31
t_p = np.linspace(0, T, B.shape[0])
lat = np.linspace(-90, 90, B.shape[1])
nm, sm = lat >= 60, lat <= -60
pinn_n, pinn_s = B[:, nm].mean(1), B[:, sm].mean(1)

s = pickle.load(open("cycle_products/store.pkl", "rb"))[21]
t_f, Bf = ct.forward(s["obs_s"][0], 0.0, s["T"], s["S"], s["t_u"])
fd_n, fd_s = ct.polar_means(Bf * ct.B_UNIT)

on, osd = ct.polar_means(s["obs"] * ct.B_UNIT)
plt.figure(figsize=(9, 4))
plt.plot(s["t_obs"], on, "k", lw=0.7, alpha=0.5, label="WSO N")
plt.plot(s["t_obs"], osd, "k--", lw=0.7, alpha=0.5, label="WSO S")
plt.plot(t_f, fd_n, "C0", label="FD N"); plt.plot(t_f, fd_s, "C0--", label="FD S")
plt.plot(t_p, pinn_n, "C1", label="PINN N"); plt.plot(t_p, pinn_s, "C1--", label="PINN S")
plt.axhline(0, color="gray", lw=0.5); plt.legend(fontsize=8, ncol=3)
plt.xlabel("years"); plt.ylabel("polar-cap mean [native units]")
plt.tight_layout(); plt.savefig("results/cycle_21/fd_pinn_overlay.png", dpi=150)