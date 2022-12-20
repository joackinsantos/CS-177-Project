# data manipulation
import pandas as pd
import numpy as np

# data visualization
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns



df = pd.read_csv("CPET_ProcessedData.csv")

print(df)


# apply some cool styling
plt.style.use("ggplot")
rcParams['figure.figsize'] = (12,  6)


# Labelling

# Collapsing dummy variables into 'diagnosis'
df.loc[df['CardiacLim'] == 1, 'diagnosis'] = 'CardiacLim'
df.loc[df['PulmonaryLim'] == 1, 'diagnosis'] = 'PulmonaryLim'
df.loc[df['MuscleSkeletalLim'] == 1, 'diagnosis'] = 'MuscleSkeletalLim'
df.loc[df['Healthy'] == 1, 'diagnosis'] = 'Healthy'

# Feature Selection

# Removing dummy variables
df = df.drop('CardiacLim', axis=1)
df = df.drop('PulmonaryLim', axis=1)
df = df.drop('MuscleSkeletalLim', axis=1)
df = df.drop('Healthy', axis=1)

# Removing IDs
df = df.drop('SessionId', axis=1)
df = df.drop('PatientId', axis=1)


# Summary

df.describe()

df.info()

df.duplicated().sum()

df['diagnosis'].value_counts()

df['diagnosis'].value_counts(normalize=True)

df['diagnosis'].value_counts().plot(kind="bar")
plt.title("Value counts of the target variable")
plt.xlabel("Diagnosis")
plt.xticks(rotation=0)
plt.ylabel("Count")
plt.show()


# Relationships of variables

sns.pairplot(df)