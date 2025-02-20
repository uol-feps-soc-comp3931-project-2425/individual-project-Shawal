import matplotlib.pyplot as plt
import pandas as pd

#load the dataset
df = pd.read_csv('eCommerce.csv')

# Count the occurrences of each detected language
language_counts = df['Language'].value_counts()

# Plot the distribution of languages
plt.figure(figsize=(8, 6))
language_counts.plot(kind='bar', color='pink')
plt.xlabel('Language')
plt.ylabel('Number of Reviews')
plt.title('Review Count by Detected Language')
plt.xticks(rotation=45)
plt.tight_layout()

# Show the plot
plt.show()
