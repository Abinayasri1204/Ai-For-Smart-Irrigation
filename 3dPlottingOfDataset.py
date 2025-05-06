import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load dataset
df = pd.read_csv("data.csv")

# Encode crops as colors
crops = df['crop'].astype('category')
colors = crops.cat.codes

# Create 3D plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

sc = ax.scatter(df['moisture'], df['temp'], df['pump'],
                c=colors, cmap='viridis', s=60)

ax.set_xlabel('Moisture')
ax.set_ylabel('Temperature')
ax.set_zlabel('Pump Status')
ax.set_title('Smart Irrigation: 3D View')

# Legend
legend_labels = crops.cat.categories
legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                    label=label, markerfacecolor=plt.cm.viridis(i / len(legend_labels)), markersize=10)
                   for i, label in enumerate(legend_labels)]
ax.legend(handles=legend_elements, title='Crop')

plt.tight_layout()
plt.show()
