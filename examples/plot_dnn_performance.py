import json
import matplotlib.pyplot as plt

with open("output/test_dnn_results.json", "r") as f_in:
    results = json.load(f_in)

fig, ax1 = plt.subplots()
ax1.set_xscale('symlog')
ax2 = ax1.twinx()

ax1.errorbar(results["lambda"], results["auc_mean"], yerr = results["auc_std"], label = "AUC", color = "red", marker = 'o')
ax2.errorbar(results["lambda"], results["p_value_mean"], yerr = results["p_value_std"], label = "p-value", color = "blue", marker = 'o')

ax1.set_xlabel(r"$\lambda$")

ax1.set_ylabel("AUC", color = "red")
ax1.tick_params(axis = "y", colors = "red")

ax2.set_ylabel(r"$\log_{10}(\mathrm{p value})$", color = "blue")
ax2.tick_params(axis = "y", colors = "blue")

fig.savefig("output/dnn_results.pdf")

