import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
import numpy as np

def load_file(file):
    data = pd.read_csv("raw_csvs/" + file, thousands=",")
    data["Date"] = pd.to_datetime(data["Date"])
    data = data.set_index("Date")
    data = data.sort_index()
    prices = pd.to_numeric(data["Price"], errors="coerce")
    return prices.dropna()

brent = load_file("LCON6.csv")

companies = {
    "Ryanair": load_file("RYA.csv"),
    "IAG": load_file("IAG.csv"),
    "Lufthansa": load_file("LHAG.csv"),
    "Air France-KLM": load_file("AIRF.csv"),
    "easyJet": load_file("EZJ.csv"),
    "Jet2": load_file("JET2.csv"),
    "Wizz Air": load_file("WIZZ.csv"),
    "Norwegian": load_file("NAS.csv"),
    "Turkish Airlines": load_file("THYAO.csv"),
    "TUI": load_file("TUI1.csv")
}

brent_returns = brent.pct_change().dropna()
brent_returns = brent_returns[abs(brent_returns) < 0.5]

results = {}

for name in companies:
    stock_returns = companies[name].pct_change().dropna()
    stock_returns = stock_returns[abs(stock_returns) < 0.5]

    dates = brent_returns.index.intersection(stock_returns.index)

    x = brent_returns.loc[dates]
    y = stock_returns.loc[dates]

    reg = linregress(x, y)

    results[name] = [
        round(reg.slope, 4),
        round(reg.rvalue, 4),
        round(reg.rvalue ** 2, 4),
        round(reg.pvalue, 4)
    ]

summary = pd.DataFrame(
    results,
    index=["Beta", "Correlation", "R2", "P-value"]
).T

summary = summary.sort_values("Beta")
print(summary)

plt.figure(figsize=(10, 5))
plt.barh(summary.index, summary["Beta"])
plt.axvline(0, color="black")
plt.title("Airline sensitivity to Brent oil returns")
plt.xlabel("Beta")
plt.tight_layout()
plt.savefig("oil_beta_chart.png")
plt.show()

for name in summary.index:
    stock_returns = companies[name].pct_change().dropna()
    stock_returns = stock_returns[abs(stock_returns) < 0.5]

    dates = brent_returns.index.intersection(stock_returns.index)
    x = brent_returns.loc[dates]
    y = stock_returns.loc[dates]

    plt.figure(figsize=(6, 4))
    plt.scatter(x, y, s=5)

    m, b = np.polyfit(x, y, 1)
    line_x = np.linspace(x.min(), x.max(), 100)
    plt.plot(line_x, m * line_x + b)

    plt.title(name)
    plt.xlabel("Brent return")
    plt.ylabel("Stock return")
    plt.tight_layout()
    plt.savefig(name.replace(" ", "_") + "_scatter.png")
    plt.show()