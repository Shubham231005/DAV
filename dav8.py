#1. Program 1- Scatter plot (create or upload any suitable data set)
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv("drinks.csv")
sns.scatterplot(
 x="beer_servings",
 y="wine_servings",
 hue="total_litres_of_pure_alcohol",
 data=df,
 palette="viridis"
)
plt.title("Beer vs Wine Servings by Alcohol Consumption")
plt.xlabel("Beer Servings")
plt.ylabel("Wine Servings")
plt.show()

#2. Program - pie chart to show cars data using Matplotlib
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("drinks.csv")
labels = ["Beer", "Spirit", "Wine"]
values = [
 df["beer_servings"].sum(),
 df["spirit_servings"].sum(),
 df["wine_servings"].sum()
]
plt.pie(values, labels=labels, autopct="%1.1f%%")
plt.title("Distribution of Drink Servings")
plt.show()
