# Pandas Lab Exercise

#### Author : Hyeri Kim

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Part 1: Setup and Basic Operations

# Importing pandas as pd for data analysis

# Creating Series objects
s = pd.Series([2, -1, 3, 5])
print(s)

# Applying NumPy functions to Series
# Using np.exp() to apply exponential function to all elements in Series 's'
exp_s = np.exp(s)
print(exp_s)

# Arithmetic operations on Series
# Adding a list to Series 's', demonstrating element-wise operations
added_series = s + [1000, 2000, 3000, 4000]
print(added_series)

# Broadcasting in Series
# Adding a scalar to all elements in Series 's'
broadcasted_series = s + 1000
print(broadcasted_series)

# Conditional operations on Series
# Checking which elements in Series 's' are less than 0
negative_elements = s < 0
print(negative_elements)

# Creating a Series with custom index labels
s2 = pd.Series([68, 83, 112, 68], index=["alice", "bob", "charles", "darwin"])
print(s2)

# Accessing Series items using labels and integer indices
# Accessing by label
bob_weight = s2.loc["bob"]
print(bob_weight)

# Accessing by integer location
second_item = s2.iloc[1]
print(second_item)

# Slicing a Series using iloc
subset_s2 = s2.iloc[1:3]
print(subset_s2)

# Handling unexpected slicing results
# Example of a surprising slicing result with default integer indexing
surprise = pd.Series([1000, 1001, 1002, 1003])
print(surprise)

# Slicing the Series
surprise_slice = surprise[2:]
print(surprise_slice)

# Attempting to access an item by default index label after slicing
try:
    print(surprise_slice[0])  # This raises a KeyError
except KeyError as e:
    print("Key error:", e)

# Correctly accessing the item using iloc
correct_access = surprise_slice.iloc[0]
print(correct_access)

# Initializing a Series from a dictionary
weights = {"alice": 68, "bob": 83, "colin": 86, "darwin": 68}
s3 = pd.Series(weights)
print(s3)

# Controlling elements and order in Series initialization
s4 = pd.Series(weights, index=["colin", "alice"])
print(s4)

# Part 2: Advanced Series Operations and Creating Custom Series

# Define a dictionary with favorite fruits and their colors
hyeri_fruits = {
    "apple": "red",
    "banana": "yellow",
    "kiwi": "green",
    "cherry": "dark red"
}

# Convert the dictionary into a pandas Series
hyeri_f = pd.Series(hyeri_fruits)
print("Original Series:")
print(hyeri_f)

# Access and print the second and third items using iloc
second_third_items = hyeri_f.iloc[1:3]
print("\nSecond and Third Items in Series:")
print(second_third_items)

# Create a sub-series from the second and third items
hyeri_f2 = hyeri_f.iloc[1:3]
print("\nSub-Series (Second and Third Items):")
print(hyeri_f2)

# Print the last item in the sub-series using iloc
last_item_sub_series = hyeri_f2.iloc[-1]
print("\nLast Item in Sub-Series:")
print(last_item_sub_series)

# Part 3: Automatic Alignment and Initialization with Scalars

# Step 1: Automatic alignment in Pandas Series
# Demonstrating automatic alignment of Series objects by index labels
print("Keys of Series s2:", s2.keys())
print("Keys of Series s3:", s3.keys())

# When adding two Series, Pandas aligns them by their index labels
aligned_sum = s2 + s3
print("\nAligned Sum of s2 and s3:")
print(aligned_sum)

# Step 2: Plotting a Series using Matplotlib
temperatures = [4.4, 5.1, 6.1, 6.2, 6.1, 6.1, 5.7, 5.2, 4.7, 4.1, 3.9, 3.5]
s7 = pd.Series(temperatures, name="Temperature")

# Plotting the temperature series
s7.plot(title="Temperature Over Time")
plt.xlabel("Index")
plt.ylabel("Temperature (°C)")
plt.show()

# Step 3: Handling time series data in Pandas
# Creating a date range for time series data
dates = pd.date_range('2016/10/29 5:30pm', periods=12, freq='H')
print("\nDatetime Index:")
print(dates)

# Creating a Series with a DatetimeIndex
temp_series = pd.Series(temperatures, index=dates)
print("\nTemperature Time Series:")
print(temp_series)

# Plotting the time series as a bar chart
temp_series.plot(kind="bar", title="Temperature Time Series")
plt.xlabel("Datetime")
plt.ylabel("Temperature (°C)")
plt.grid(True)
plt.show()

# Step 4: Working with Rainfall Data as a Time Series
hyeri_amounts = [10, 23, 24, 30]  # Rainfall amounts
hyeri_date = pd.date_range("2023/2/5 2:00pm", periods=4, freq='H')  # Creating a time range

# Creating a pandas series for rainfall amounts
hyeri_rainfall_amount_today = pd.Series(hyeri_amounts, index=hyeri_date)
print("\nHyeri Rainfall Amounts Today:")
print(hyeri_rainfall_amount_today)

# Plotting the rainfall data as a bar chart
hyeri_rainfall_amount_today.plot(kind="bar", title="Rainfall Amounts Today")
plt.xlabel("Time")
plt.ylabel("Rainfall Amount (mm)")
plt.show()

# Step 5: Resampling Time Series Data
# Resampling the temperature data to a 2-hour frequency and calculating the mean
temp_series_freq_2H = temp_series.resample("2H").mean()
print("\nResampled Temperature (2-Hour Frequency, Mean):")
print(temp_series_freq_2H)

# Plotting the resampled data
temp_series_freq_2H.plot(kind="bar", title="Resampled Temperature (2-Hour Mean)")
plt.xlabel("Datetime")
plt.ylabel("Temperature (°C)")
plt.show()

# Resampling using the minimum value for each 2-hour period
temp_series_freq_2H_min = temp_series.resample("2H").min()
print("\nResampled Temperature (2-Hour Frequency, Min):")
print(temp_series_freq_2H_min)

# Applying a custom aggregation function (min) with resample
temp_series_freq_2H_custom = temp_series.resample("2H").apply(np.min)
print("\nResampled Temperature (Custom Aggregation - Min):")
print(temp_series_freq_2H_custom)

# Part 4: Upsampling, Interpolation, Timezones, and Period Handling

# Step 1: Upsampling and Interpolation of Time Series Data

# Upsampling the temperature series to a 15-minute frequency, initially creates gaps
temp_series_freq_15min = temp_series.resample("15Min").mean()
print("\nUpsampled Temperature (15-Minute Frequency, Mean):")
print(temp_series_freq_15min.head(n=10))  # Display the first 10 entries

# Filling gaps using cubic interpolation
temp_series_freq_15min = temp_series.resample("15Min").interpolate(method="cubic")
print("\nUpsampled Temperature (15-Minute Frequency, Cubic Interpolation):")
print(temp_series_freq_15min.head(n=10))  # Display the first 10 interpolated entries

# Plotting the original and interpolated series for comparison
temp_series.plot(label="Period: 1 hour", title="Temperature Time Series with Interpolation")
temp_series_freq_15min.plot(label="Period: 15 minutes")
plt.xlabel("Datetime")
plt.ylabel("Temperature (°C)")
plt.legend()
plt.show()

# Step 2: Handling Timezones in Time Series Data

# Localizing the time series to New York timezone
temp_series_ny = temp_series.tz_localize("America/New_York")
print("\nTime Series in New York Timezone:")
print(temp_series_ny)

# Converting the time series to Paris timezone
temp_series_paris = temp_series_ny.tz_convert("Europe/Paris")
print("\nTime Series Converted to Paris Timezone:")
print(temp_series_paris)

# Removing timezone information to create naive datetime objects
temp_series_paris_naive = temp_series_paris.tz_localize(None)
print("\nNaive Time Series (No Timezone):")
print(temp_series_paris_naive)

# Attempting to re-localize to Paris timezone, expecting an ambiguity error
try:
    temp_series_paris_naive.tz_localize("Europe/Paris")
except Exception as e:
    print("\nError when localizing naive time series:")
    print(type(e), e)

# Resolving ambiguity by inferring correct timezone conversion
temp_series_paris_localized = temp_series_paris_naive.tz_localize("Europe/Paris", ambiguous="infer")
print("\nTime Series with Resolved Ambiguity (Paris Timezone):")
print(temp_series_paris_localized)

# Step 3: Handling Period Data in Pandas

# Creating a PeriodIndex for quarterly data
quarters = pd.period_range('2016Q1', periods=8, freq='Q')
print("\nQuarterly Period Index:")
print(quarters)

# Shifting periods by adding an integer
shifted_quarters = quarters + 3
print("\nShifted Quarterly Periods by 3:")
print(shifted_quarters)

# Converting quarterly periods to monthly periods
monthly_quarters_end = quarters.asfreq("M")
print("\nConverted to Monthly Periods (End of Quarter):")
print(monthly_quarters_end)

# Converting to the start of each period
monthly_quarters_start = quarters.asfreq("M", how="start")
print("\nConverted to Monthly Periods (Start of Quarter):")
print(monthly_quarters_start)

# Converting to annual frequency
annual_quarters = quarters.asfreq("A")
print("\nConverted to Annual Frequency:")
print(annual_quarters)

# Creating a time series with quarterly revenue
quarterly_revenue = pd.Series([300, 320, 290, 390, 320, 360, 310, 410], index=quarters)
print("\nQuarterly Revenue Series:")
print(quarterly_revenue)

# Plotting the quarterly revenue series
quarterly_revenue.plot(kind="line", title="Quarterly Revenue Over Time")
plt.xlabel("Quarter")
plt.ylabel("Revenue")
plt.show()


# Handling Timezones: Making Datetimes Timezone Aware
# Localizing the time series to New York timezone
temp_series_ny = temp_series.tz_localize("America/New_York")
print(temp_series_ny)

# Converting the time series to Paris timezone
temp_series_paris = temp_series_ny.tz_convert("Europe/Paris")
print(temp_series_paris)

# Periods: Creating and Manipulating Period Index
quarters = pd.period_range('2016Q1', periods=8, freq='Q')  # Creating a period range
print(quarters)

# Creating a quarterly revenue Series and plotting it
quarterly_revenue = pd.Series([300, 320, 290, 390, 320, 360, 310, 410], index=quarters)
quarterly_revenue.plot(kind="line")
plt.show()

# Part 5: DataFrame Creation, Multi-Indexing, Stacking/Unstacking, and Operations

# Step 1: Creating DataFrames in Pandas

# Creating a DataFrame using a dictionary of Series
people_dict = {
    "weight": pd.Series([68, 83, 112], index=["alice", "bob", "charles"]),
    "birthyear": pd.Series([1984, 1985, 1992], index=["bob", "alice", "charles"], name="year"),
    "children": pd.Series([0, 3], index=["charles", "bob"]),
    "hobby": pd.Series(["Biking", "Dancing"], index=["alice", "bob"]),
}
people = pd.DataFrame(people_dict)
print("\nPeople DataFrame:")
print(people)

# Step 2: Working with Multi-Index DataFrames

# Creating a multi-index DataFrame
d5 = pd.DataFrame(
    {
        ("public", "birthyear"): {("Paris", "alice"): 1985, ("Paris", "bob"): 1984, ("London", "charles"): 1992},
        ("public", "hobby"): {("Paris", "alice"): "Biking", ("Paris", "bob"): "Dancing"},
        ("private", "weight"): {("Paris", "alice"): 68, ("Paris", "bob"): 83, ("London", "charles"): 112},
        ("private", "children"): {("Paris", "alice"): np.nan, ("Paris", "bob"): 3, ("London", "charles"): 0}
    }
)
print("\nMulti-Index DataFrame (d5):")
print(d5)

# Creating a copy of the DataFrame to avoid modifying the original
hyeri_d5 = d5.copy()

# Step 3: Selecting Specific Columns from Multi-Index DataFrame
# Extracting all "private" columns from the multi-index DataFrame
hyeri_d5_private = hyeri_d5["private"]
print("\nPrivate Columns DataFrame:")
print(hyeri_d5_private)

# Transposing the DataFrame to swap rows and columns
hyeri_d5_private_transposed = hyeri_d5_private.T
print("\nTransposed Private Columns DataFrame:")
print(hyeri_d5_private_transposed)

# Step 4: Dropping a Level in Multi-Index Columns
# Dropping the first level of the column index
d5.columns = d5.columns.droplevel(level=0)
print("\nDataFrame after Dropping Column Level:")
print(d5)

# Step 5: Transposing, Stacking, and Unstacking DataFrames

# Transposing the DataFrame to swap rows and columns
d6 = d5.T
print("\nTransposed DataFrame (d6):")
print(d6)

# Stacking the DataFrame: Converting columns to a new level of rows
d7 = d6.stack()
print("\nStacked DataFrame (d7):")
print(d7)

# Unstacking the DataFrame: Converting a level of rows back to columns
d8 = d7.unstack()
print("\nUnstacked DataFrame (d8):")
print(d8)

# Further unstacking levels in the DataFrame
d9 = d8.unstack()
print("\nDouble Unstacked DataFrame (d9):")
print(d9)

# Unstacking multiple levels at once
d10 = d9.unstack(level=(0, 1))
print("\nDataFrame after Unstacking Multiple Levels (d10):")
print(d10)

# Step 6: Accessing Rows in DataFrames

# Accessing rows using loc (by label) and iloc (by integer location)
charles_info = people.loc["charles"]
print("\nInformation about Charles:")
print(charles_info)

second_row_info = people.iloc[2]
print("\nInformation in the Second Row:")
print(second_row_info)

slice_of_rows = people.iloc[1:3]
print("\nSlice of Rows (1 to 3):")
print(slice_of_rows)

# Filtering rows based on a condition
people_born_before_1990 = people[people["birthyear"] < 1990]
print("\nPeople Born Before 1990:")
print(people_born_before_1990)

# Step 7: Adding and Removing Columns in DataFrames

# Adding new columns to the DataFrame
people["age"] = 2018 - people["birthyear"]  # Adding an 'age' column
people["over 30"] = people["age"] > 30  # Adding a column for people over 30
print("\nPeople DataFrame After Adding Columns:")
print(people)

# Removing columns from the DataFrame
birthyears = people.pop("birthyear")  # Removing 'birthyear' column and storing it
del people["children"]  # Deleting the 'children' column
print("\nPeople DataFrame After Removing Columns:")
print(people)
print("\nExtracted Birthyears:")
print(birthyears)

# Step 8: Modifying DataFrame Copy with New Column

# Creating a copy of the DataFrame and modifying it
hyeri_people = people.copy()
hyeri_people["education"] = pd.Series({"alice": "Diploma", "bob": "Masters"})
print("\nModified DataFrame (hyeri_people):")
print(hyeri_people)

# Printing all information related to 'alice'
alice_info = hyeri_people.loc["alice"]
print("\nInformation Related to Alice:")
print

# Displaying Alice's information again as a DataFrame
alice_info_df = hyeri_people.iloc[0:1]
print("\nAlice's Information as DataFrame:")
print(alice_info_df)

# Part 6: Assigning New Columns, Evaluating Expressions, Querying, Sorting, and Plotting DataFrames

# Step 1: Assigning New Columns Using the assign() Method

# Adding multiple new columns to the DataFrame using assign()
people = people.assign(
    body_mass_index = people["weight"] / (people["height"] / 100) ** 2,  # Calculating BMI
    has_pets = people["pets"] > 0  # Checking if people have pets
)
print("\nPeople DataFrame After Assigning New Columns (body_mass_index, has_pets):")
print(people)

# Step 2: Evaluating Expressions with eval()

# Evaluating an expression to identify if the BMI is over 25
people["is_overweight"] = people.eval("weight / (height/100) ** 2 > 25")
print("\nPeople DataFrame After Evaluating Overweight Condition:")
print(people)

# Creating new columns using eval() method with inplace modification
people.eval("body_mass_index = weight / (height/100) ** 2", inplace=True)  # Adding body_mass_index column
people.eval("overweight = body_mass_index > @overweight_threshold", inplace=True)  # Checking if BMI is over the threshold
print("\nPeople DataFrame After eval() Assignments:")
print(people)

# Step 3: Querying DataFrames

# Using query() method to filter DataFrame based on conditions
people_over_30_no_pets = people.query("age > 30 and pets == 0")
print("\nPeople Over 30 Without Pets:")
print(people_over_30_no_pets)

# Querying the custom DataFrame created earlier
hyeri_people_under_180 = hyeri_people.query("height < 180")
print("\nHyeri People Under 180cm in Height:")
print(hyeri_people_under_180)

# Step 4: Sorting DataFrames

# Sorting the DataFrame by index in descending order
people_sorted_by_index = people.sort_index(ascending=False)
print("\nPeople DataFrame Sorted by Index (Descending):")
print(people_sorted_by_index)

# Sorting the DataFrame by column names
people.sort_index(axis=1, inplace=True)  # Sort columns alphabetically in place
print("\nPeople DataFrame After Sorting Columns by Name:")
print(people)

# Sorting the DataFrame by values in the 'age' column
people.sort_values(by="age", inplace=True)
print("\nPeople DataFrame After Sorting by Age:")
print(people)

# Step 5: Plotting DataFrames

# Plotting a line chart for body mass index against height and weight
people.plot(kind="line", x="body_mass_index", y=["height", "weight"], title="BMI vs. Height and Weight")
plt.xlabel("Body Mass Index")
plt.ylabel("Values")
plt.show()

# Plotting a scatter plot for height vs. weight with custom sizes
people.plot(kind="scatter", x="height", y="weight", s=[40, 120, 200], title="Height vs. Weight Scatter Plot")
plt.xlabel("Height")
plt.ylabel("Weight")
plt.show()

# Part 7: Operations on DataFrames, Handling Missing Data, Grouping, and Pivot Tables

# Step 1: Operations on DataFrames

# Creating a DataFrame for grades
grades_array = np.array([[8, 8, 9], [10, 9, 9], [4, 8, 2], [9, 10, 10]])
grades = pd.DataFrame(grades_array, columns=["sep", "oct", "nov"], index=["alice", "bob", "charles", "darwin"])
print("\nGrades DataFrame:")
print(grades)

# Applying NumPy mathematical functions on DataFrame
grades_sqrt = np.sqrt(grades)
print("\nSquare Root of Grades DataFrame:")
print(grades_sqrt)

# Adding a scalar to all elements in DataFrame (broadcasting)
grades_plus_one = grades + 1
print("\nGrades DataFrame After Adding 1:")
print(grades_plus_one)

# Conditional checks on DataFrame elements
grades_greater_equal_five = grades >= 5
print("\nGrades Greater or Equal to 5:")
print(grades_greater_equal_five)

# Calculating the mean for each column
grades_mean = grades.mean()
print("\nMean Grades for Each Month:")
print(grades_mean)

# Checking if all grades are greater than 5 for each column
all_grades_greater_five = (grades > 5).all()
print("\nAll Grades Greater Than 5 for Each Month:")
print(all_grades_greater_five)

# Checking if all grades are greater than 5 for each row
all_grades_greater_five_by_row = (grades > 5).all(axis=1)
print("\nAll Grades Greater Than 5 for Each Student:")
print(all_grades_greater_five_by_row)

# Checking if any grade is 10 for each row
any_grade_is_ten = (grades == 10).any(axis=1)
print("\nAny Grade is 10 for Each Student:")
print(any_grade_is_ten)

# Subtracting the column means from each element
grades_minus_mean = grades - grades.mean()
print("\nGrades DataFrame After Subtracting Column Means:")
print(grades_minus_mean)

# Subtracting the global mean from all elements
grades_minus_global_mean = grades - grades.values.mean()
print("\nGrades DataFrame After Subtracting Global Mean:")
print(grades_minus_global_mean)

# Step 2: Handling Missing Data

# Creating a DataFrame for bonus points with missing data
bonus_array = np.array([[0, np.nan, 2], [np.nan, 1, 0], [0, 1, 0], [3, 3, 0]])
bonus_points = pd.DataFrame(bonus_array, columns=["oct", "nov", "dec"], index=["bob", "colin", "darwin", "charles"])
print("\nBonus Points DataFrame with Missing Data:")
print(bonus_points)

# Filling missing data with zeros
filled_bonus_points = (grades + bonus_points).fillna(0)
print("\nGrades After Adding Bonus Points and Filling Missing Data with 0:")
print(filled_bonus_points)

# Interpolating missing data horizontally
interpolated_bonus_points = bonus_points.interpolate(axis=1)
print("\nInterpolated Bonus Points DataFrame:")
print(interpolated_bonus_points)

# Combining grades and bonus points with filled data
grades["dec"] = np.nan  # Adding a new column with NaN values
final_grades = grades + bonus_points.fillna(0)
print("\nFinal Grades After Adding Bonus Points:")
print(final_grades)

# Dropping rows and columns where all values are NaN
final_grades_cleaned = final_grades.dropna(how="all").dropna(axis=1, how="all")
print("\nFinal Grades After Dropping Rows and Columns Full of NaN:")
print(final_grades_cleaned)

# Step 3: Grouping and Pivot Tables

# Adding a categorical column to the DataFrame
final_grades["hobby"] = ["Biking", "Dancing", np.nan, "Dancing", "Biking"]
print("\nFinal Grades with Hobby Column:")
print(final_grades)

# Grouping the DataFrame by the 'hobby' column
grouped_grades = final_grades.groupby("hobby")
print("\nGrouped Grades by Hobby:")
print(grouped_grades.mean())  # Calculating mean grades for each group

# Creating a Pivot Table
pivot_by_name = pd.pivot_table(more_grades, index="name")
print("\nPivot Table by Name:")
print(pivot_by_name)

# Pivot table with specific aggregation function
pivot_max_grade = pd.pivot_table(more_grades, index="name", values=["grade", "bonus"], aggfunc=np.max)
print("\nPivot Table with Maximum Grades and Bonus Points:")
print(pivot_max_grade)

# Pivot table with multiple levels and margins
pivot_with_margins = pd.pivot_table(more_grades, index="name", values="grade", columns="month", margins=True)
print("\nPivot Table with Grades, Months, and Margins:")
print(pivot_with_margins)

# Pivot table with multi-level indices
multi_level_pivot = pd.pivot_table(more_grades, index=("name", "month"), margins=True)
print("\nMulti-Level Pivot Table:")
print(multi_level_pivot)

# Part 8: Combining DataFrames, SQL-like Joins, Concatenation, and Categorical Data

# Step 1: Combining DataFrames Using SQL-like Joins

# Creating a DataFrame with city locations
city_loc = pd.DataFrame(
    [
        ["CA", "San Francisco", 37.781334, -122.416728],
        ["NY", "New York", 40.705649, -74.008344],
        ["FL", "Miami", 25.791100, -80.320733],
        ["OH", "Cleveland", 41.473508, -81.739791],
        ["UT", "Salt Lake City", 40.755851, -111.896657]
    ], columns=["state", "city", "lat", "lng"]
)
print("\nCity Locations DataFrame:")
print(city_loc)

# Creating a DataFrame with city populations
city_pop = pd.DataFrame(
    [
        [808976, "San Francisco", "California"],
        [8363710, "New York", "New-York"],
        [413201, "Miami", "Florida"],
        [2242193, "Houston", "Texas"]
    ], index=[3, 4, 5, 6], columns=["population", "city", "state"]
)
print("\nCity Populations DataFrame:")
print(city_pop)

# Performing an inner join to combine DataFrames on the "city" column
inner_join = pd.merge(left=city_loc, right=city_pop, on="city")
print("\nInner Join on 'city' Column:")
print(inner_join)

# Performing a full outer join to include all records from both DataFrames
outer_join = pd.merge(left=city_loc, right=city_pop, on="city", how="outer")
print("\nFull Outer Join on 'city' Column:")
print(outer_join)

# Performing a right outer join to include all records from the right DataFrame
right_join = pd.merge(left=city_loc, right=city_pop, on="city", how="right")
print("\nRight Join on 'city' Column:")
print(right_join)

# Step 2: Handling Joins with Different Key Names

# Renaming columns in the city population DataFrame
city_pop2 = city_pop.copy()
city_pop2.columns = ["population", "name", "state"]

# Merging DataFrames with different key names using 'left_on' and 'right_on'
custom_key_merge = pd.merge(left=city_loc, right=city_pop2, left_on="city", right_on="name")
print("\nCustom Key Merge Between city_loc and city_pop2:")
print(custom_key_merge)

# Step 3: Concatenating DataFrames

# Concatenating DataFrames vertically (default) and resetting index
vertical_concat = pd.concat([city_loc, city_pop], ignore_index=True)
print("\nVertical Concatenation of DataFrames (With Reset Index):")
print(vertical_concat)

# Concatenating DataFrames with inner join, retaining only common columns
inner_concat = pd.concat([city_loc, city_pop], join="inner")
print("\nInner Concatenation of DataFrames (Common Columns Only):")
print(inner_concat)

# Concatenating DataFrames horizontally (axis=1)
horizontal_concat = pd.concat([city_loc.set_index("city"), city_pop.set_index("city")], axis=1)
print("\nHorizontal Concatenation of DataFrames (Indexed by 'city'):")
print(horizontal_concat)

# Using append() method as shorthand for vertical concatenation
appended_df = city_loc.append(city_pop, ignore_index=True)
print("\nAppended DataFrame (Equivalent to Vertical Concatenation):")
print(appended_df)

# Step 4: Working with Categorical Data

# Adding a categorical column to the DataFrame
city_eco = city_pop.copy()
city_eco["eco_code"] = [17, 17, 34, 20]
print("\nCity Economic DataFrame with Categorical Codes:")
print(city_eco)

# Converting the 'eco_code' column to a categorical data type
city_eco["economy"] = city_eco["eco_code"].astype('category')
print("\nCategories in 'economy' Column (Before Renaming):")
print(city_eco["economy"].cat.categories)

# Renaming the categories to meaningful names
city_eco["economy"].cat.categories = ["Finance", "Energy", "Tourism"]
print("\nCity Economic DataFrame After Renaming Categories:")
print(city_eco)

# Sorting the DataFrame based on the categorical column
sorted_city_eco = city_eco.sort_values(by="economy", ascending=False)
print("\nCity Economic DataFrame Sorted by 'economy' Column:")
print(sorted_city_eco)

# Part 9: Saving and Loading Data, Combining DataFrames, and Handling Large DataFrames

# Step 1: Saving and Loading DataFrames

# Creating a DataFrame with sample data
my_df = pd.DataFrame(
    [["Biking", 68.5, 1985, np.nan], ["Dancing", 83.1, 1984, 3]],
    columns=["hobby", "weight", "birthyear", "children"],
    index=["alice", "bob"]
)
print("\nOriginal DataFrame (my_df):")
print(my_df)

# Saving the DataFrame to various file formats
my_df.to_csv("my_df.csv")   # Saving as CSV
my_df.to_html("my_df.html")  # Saving as HTML
my_df.to_json("my_df.json")  # Saving as JSON

# Displaying saved file contents
for filename in ("my_df.csv", "my_df.html", "my_df.json"):
    print(f"\nContents of {filename}:")
    with open(filename, "rt") as f:
        print(f.read())

# Loading the CSV file back into a DataFrame
my_df_loaded = pd.read_csv("my_df.csv", index_col=0)
print("\nLoaded DataFrame from CSV:")
print(my_df_loaded)

# Step 2: Combining DataFrames

# Creating DataFrames for city location and population
city_loc = pd.DataFrame(
    [
        ["CA", "San Francisco", 37.781334, -122.416728],
        ["NY", "New York", 40.705649, -74.008344],
        ["FL", "Miami", 25.791100, -80.320733],
        ["OH", "Cleveland", 41.473508, -81.739791],
        ["UT", "Salt Lake City", 40.755851, -111.896657]
    ], columns=["state", "city", "lat", "lng"]
)
print("\nCity Locations DataFrame:")
print(city_loc)

city_pop = pd.DataFrame(
    [
        [808976, "San Francisco", "California"],
        [8363710, "New York", "New-York"],
        [413201, "Miami", "Florida"],
        [2242193, "Houston", "Texas"]
    ], index=[3, 4, 5, 6], columns=["population", "city", "state"]
)
print("\nCity Populations DataFrame:")
print(city_pop)

# Concatenating DataFrames vertically (default axis=0)
concatenated_df = pd.concat([city_loc, city_pop])
print("\nVertically Concatenated DataFrame:")
print(concatenated_df)

# Concatenating DataFrames horizontally by setting 'city' as the index
horizontal_concat_df = pd.concat([city_loc.set_index("city"), city_pop.set_index("city")], axis=1)
print("\nHorizontally Concatenated DataFrame (Indexed by 'city'):")
print(horizontal_concat_df)

# Step 3: Handling Large DataFrames

# Creating a large DataFrame using a mathematical function
much_data = np.fromfunction(lambda x, y: (x + y * y) % 17 * 11, (10000, 26))
large_df = pd.DataFrame(much_data, columns=list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))

# Replacing certain values with NaN to simulate missing data
large_df[large_df % 16 == 0] = np.nan

# Inserting a column with repeated text data to simulate mixed data types
large_df.insert(3, "some_text", "Blabla")
print("\nLarge DataFrame Sample (First 5 Rows):")
print(large_df.head())

# Viewing the first few rows to get an overview of the DataFrame
print("\nFirst 5 Rows of Large DataFrame:")
print(large_df.head())

# Viewing the last 2 rows of the DataFrame
print("\nLast 2 Rows of Large DataFrame:")
print(large_df.tail(n=2))

# Getting an overview of the DataFrame's content and structure
print("\nInformation About Large DataFrame:")
large_df.info()

# Getting descriptive statistics for numeric columns
print("\nDescriptive Statistics for Large DataFrame:")
print(large_df.describe())


