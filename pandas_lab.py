# Pandas Lab Exercise

#### Author : Hyeri Kim

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Part 1: Setup and Basic Operations

# Creating Series objects
s = pd.Series([2, -1, 3, 5])
print("Series s:\n", s.to_string(), sep='')

# Applying NumPy functions to Series
exp_s = np.exp(s)
print("Exponential of Series s:\n", exp_s, sep='')

# Arithmetic operations on Series
added_series = s + [1000, 2000, 3000, 4000]
print("Series s + [1000, 2000, 3000, 4000]:\n", added_series.to_string(), sep='')

# Broadcasting in Series
broadcasted_series = s + 1000
print("Series s + 1000:\n", broadcasted_series.to_string(), sep='')

# Conditional operations on Series
negative_elements = s < 0
print("Elements in Series s < 0:\n", negative_elements.to_string(), sep='')

# Creating a Series with custom index labels
s2 = pd.Series([68, 83, 112, 68], index=["alice", "bob", "charles", "darwin"])
print("Series s2 with custom index labels:\n", s2.to_string(), sep='')

# Accessing Series items using labels and integer indices
bob_weight = s2.loc["bob"]
print("Weight of Bob (using loc):", bob_weight)

second_item = s2.iloc[1]
print("Second item in Series s2 (using iloc):", second_item)

# Part 2: Creating DataFrames and Basic Operations

# Creating a DataFrame using a dictionary of Series
people_dict = {
    "weight": pd.Series([68, 83, 112], index=["alice", "bob", "charles"]),
    "birthyear": pd.Series([1984, 1985, 1992], index=["bob", "alice", "charles"]),
    "children": pd.Series([0, 3], index=["charles", "bob"]),
    "hobby": pd.Series(["Biking", "Dancing"], index=["alice", "bob"]),
}
people = pd.DataFrame(people_dict)
print("People DataFrame:\n", people.to_string(), sep='')

# Adding new columns
people["age"] = 2023 - people["birthyear"]
people["over 30"] = people["age"] > 30
print("People DataFrame After Adding Columns:\n", people.to_string(), sep='')

# Removing columns
birthyears = people.pop("birthyear")
del people["children"]
print("People DataFrame After Removing Columns:\n", people.to_string(), sep='')
print("Extracted Birthyears:\n", birthyears.to_string(), sep='')

# Part 3: Handling Duplicates and Data Cleaning

# Creating a DataFrame with duplicates
data_with_duplicates = pd.DataFrame({
    "Name": ["Alice", "Bob", "Charlie", "Bob", "Alice"],
    "Age": [25, 30, 35, 30, 25],
    "City": ["New York", "Paris", "London", "Paris", "New York"]
})
print("DataFrame with Duplicates:\n", data_with_duplicates.to_string(), sep='')

# Identifying duplicates
duplicates = data_with_duplicates.duplicated()
print("Identifying Duplicate Rows:\n", duplicates.to_string(), sep='')

# Removing duplicates
data_no_duplicates = data_with_duplicates.drop_duplicates()
print("DataFrame After Removing Duplicates:\n", data_no_duplicates.to_string(), sep='')

# Part 4: String Operations on Series and DataFrames

# Creating a Series of names
names = pd.Series(["alice", "BOB", "Charlie", "david", "Eva"])
print("Original Names Series:\n", names.to_string(), sep='')

# Converting to lowercase and uppercase
names_lower = names.str.lower()
names_upper = names.str.upper()
print("Names in Lowercase:\n", names_lower.to_string(), sep='')
print("Names in Uppercase:\n", names_upper.to_string(), sep='')

# Stripping whitespace and capitalizing
names_cleaned = names.str.strip().str.capitalize()
print("Cleaned Names Series:\n", names_cleaned.to_string(), sep='')

# Applying string operations on DataFrame columns
people["hobby"] = people["hobby"].str.upper()
print("People DataFrame After Applying String Operations on 'hobby' Column:\n", people.to_string(), sep='')

# Part 5: Grouping Data and Aggregation

# Creating a DataFrame for demonstration
group_data = pd.DataFrame({
    "Department": ["HR", "Finance", "HR", "Finance", "HR", "Finance"],
    "Employee": ["John", "Emily", "Anna", "Tom", "Steve", "Sarah"],
    "Salary": [50000, 60000, 70000, 80000, 65000, 90000]
})
print("Employee Salary DataFrame:\n", group_data.to_string(), sep='')

# Grouping data by department and calculating the mean salary
mean_salary_by_department = group_data.groupby("Department")["Salary"].mean()
print("Mean Salary by Department:\n", mean_salary_by_department.to_string(), sep='')

# Grouping data by multiple columns and applying aggregation functions
grouped_agg = group_data.groupby(["Department", "Employee"]).agg({"Salary": ["mean", "sum"]})
print("Grouped Data with Multiple Aggregation Functions:\n", grouped_agg.to_string(), sep='')

# Part 6: Working with Dates and Times

# Creating a date range for time series data
date_range = pd.date_range('2022-01-01', periods=10, freq='D')
print("Date Range:\n", date_range.to_series().to_string(), sep='')

# Creating a DataFrame with a DateTimeIndex
date_df = pd.DataFrame({
    "Sales": np.random.randint(100, 200, size=10)
}, index=date_range)
print("DateTime Indexed DataFrame:\n", date_df.to_string(), sep='')

# Extracting parts of the datetime
date_df["Year"] = date_df.index.year
date_df["Month"] = date_df.index.month
date_df["Day"] = date_df.index.day
print("DataFrame with Extracted Date Parts:\n", date_df.to_string(), sep='')

# Calculating date differences
date_diff = date_df.index.to_series().diff().dt.days
print("Day Differences Between Consecutive Dates:\n", date_diff.to_string(), sep='')

# Part 7: Merging and Joining DataFrames

# Creating a DataFrame with city location data
city_loc = pd.DataFrame(
    [
        ["CA", "San Francisco", 37.781334, -122.416728],
        ["NY", "New York", 40.705649, -74.008344],
        ["FL", "Miami", 25.791100, -80.320733],
        ["OH", "Cleveland", 41.473508, -81.739791],
        ["UT", "Salt Lake City", 40.755851, -111.896657]
    ], columns=["state", "city", "lat", "lng"]
)
print("City Locations DataFrame:\n", city_loc.to_string(), sep='')

# Creating a DataFrame with city populations
city_pop = pd.DataFrame(
    [
        [808976, "San Francisco", "California"],
        [8363710, "New York", "New-York"],
        [413201, "Miami", "Florida"],
        [2242193, "Houston", "Texas"]
    ], index=[3, 4, 5, 6], columns=["population", "city", "state"]
)
print("City Populations DataFrame:\n", city_pop.to_string(), sep='')

# Merging city location and city information DataFrames
merged_city_data = pd.merge(city_loc, city_pop, on="city", how="left")
print("Merged City DataFrame with City Information:\n", merged_city_data.to_string(), sep='')

# Part 8: Saving and Loading DataFrames

# Creating a DataFrame with sample data
my_df_pandas_lab = pd.DataFrame(
    [["Biking", 68.5, 1985, np.nan], ["Dancing", 83.1, 1984, 3]],
    columns=["hobby", "weight", "birthyear", "children"],
    index=["alice", "bob"]
)
print("Original DataFrame (my_df):\n", my_df_pandas_lab.to_string(), sep='')

# Saving the DataFrame to various file formats
my_df_pandas_lab.to_csv("my_df_pandas_lab.csv")   # Saving as CSV
my_df_pandas_lab.to_json("my_df_pandas_lab.json")  # Saving as JSON

# Loading the CSV file back into a DataFrame
my_df_pandas_lab_loaded = pd.read_csv("my_df_pandas_lab.csv", index_col=0)
print("Loaded DataFrame from CSV:\n", my_df_pandas_lab_loaded.to_string(), sep='')

