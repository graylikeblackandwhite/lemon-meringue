import matplotlib.pyplot as plt

# Data
labels = ['Control', 'α=0.6, β=0.4', 'α=1.2, β=0.4']
values = [650, 469.7, 472.4]

# Plot
plt.figure(figsize=(7, 5))
bars = plt.bar(labels, values, color=['gray', 'skyblue', 'orange'])
plt.ylabel('Energy (kWh)')
plt.title('Energy Consumption Comparison')
plt.ylim(0, 700)

# Annotate bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 10, f'{yval:.1f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()