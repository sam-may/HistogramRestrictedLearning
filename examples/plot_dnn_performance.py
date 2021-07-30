import json
import matplotlib.pyplot as plt

with open("output/test_dnn_results_ks_d_value.json") as f_in:
    results = json.load(f_in)

fig, ax1 = plt.subplots()
ax1.set_xscale('symlog', linthreshx = 0.001)
ax2 = ax1.twinx()

ax1.errorbar(results["lambda"], results["auc_mean"], yerr = results["auc_std"], label = "AUC", color = "red", marker = 'o')
ax2.errorbar(results["lambda"], results["p_value_mean"], yerr = results["p_value_std"], label = "p-value", color = "blue", marker = 'o')

ax1.set_xlabel(r"$\lambda$")

ax1.set_ylabel("AUC", color = "red")
ax1.tick_params(axis = "y", colors = "red")

ax2.set_ylabel(r"$\log_{10}(\mathrm{p value})$", color = "blue")
ax2.tick_params(axis = "y", colors = "blue")

ax1.set_ylim([0.5, 1.0])
ax2.set_ylim([-15, 0])

fig.savefig("output/dnn_results_ks_d_value.pdf")

