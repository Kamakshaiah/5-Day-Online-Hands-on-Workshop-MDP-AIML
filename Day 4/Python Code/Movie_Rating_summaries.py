```python
print(f"Shape: {df.shape}")  # This prints out (50, 6) confirming that we indeed have 50 rows and 6 columns in our dataframe 'df'
print("Column names are as follows:\n", df.columns)  # Outputs the column list you provided ['UserID', 'User Name', 'ItemID', 'Movie Name', 'Rating', 'Timestamp'] to confirm their presence
```


```python
print("First five entries in our dataframe:\n", df.head())  # This will show the first 5 rows of your data for quick visualization and verification purposes  
print("\nSome information about each column like count, unique values etc.\n")
print(df.info())  # Provides a concise summary including important features such as datatype and memory usage of dataframe entries
```


```python
import seaborn as sns

print("\nCreating plot that displays how many times users have rated every Movie Name, it gives insight into popularity and user engagement.\n")
movie_rating_counts = df['Movie Name'].value_counts().head(10)  # Get the top 10 movies based on rating count for visualization. Modify this value to display other amounts of top rated movies if required  
top_movies = movie_rating_counts.index                          # These are names (for plot x-axis labels, which we will use)

```
