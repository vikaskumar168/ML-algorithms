import pandas as pd
data = pd.read_csv('president_heights.csv')
print("Dataset loaded successfully.")
print(data.head())

import numpy as np
height = np.array(data['height(cm)'])
print(f"Extracted heights: {height}")

print("Mean height:", height.mean())
print("Minimum height:", height.min())
print("Maximum height:", height.max())
print("Standard Deviation of height:", height.std())


print("25th Percentile:", np.percentile(height, 25))
print("Median:", np.median(height))
print("75th Percentile:", np.percentile(height, 75))


import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

plt.hist(height)
plt.title("Distribution of Presidential Heights")
plt.xlabel("Height (cm)")
plt.ylabel("Number of Presidents")  
plt.savefig("presidential_heights_distribution.png")


#more detailed visualization
plt.clf()
sns.histplot(height, bins=10, kde=True)
plt.title("Detailed Distribution of Presidential Heights")
plt.xlabel("Height (cm)")
plt.ylabel("Number of Presidents")
plt.savefig("detailed_presidential_heights_distribution.png")


#box plot
plt.clf()
sns.boxplot(y=height)
plt.title("Box Plot of Presidential Heights")
plt.ylabel("Height (cm)")
plt.savefig("presidential_heights_boxplot.png")


#violin plot
plt.clf()
sns.violinplot(y=height)
plt.title("Violin Plot of Presidential Heights")
plt.ylabel("Height (cm)")
plt.savefig("presidential_heights_violinplot.png")
