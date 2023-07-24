# Read in files
import pandas as pd

# File is examples_same_steps.csv
df_same = pd.read_csv('../ball_drop/ball_drop.csv')

# Want to convert the dataframes to lines of text
# An object is dropped at time {t_0} with initial position {x_0}, {y_0} and initial velocity {dx_0}, {dy_0}.
# Gravity is {GRAVITY[1]}. What is the position of the vertical object at time {t_1}?  Answer: {y_1}

GRAVITY = (0, -0.1)

# print the first 5 rows of the dataframe
print(df_same.head())
# Generate text for same steps
with open('../ball_drop/examples_ball_drop_min.txt', 'w') as f:
    for _, row in df_same.iterrows():
        f.write(
            f"ball_1 - y1: {row['y1']}; v1: {row['dy1']};\nball_2 - y2: {row['y2']}; v2: {row['dy2']};\n")
        f.write(f"Answer: {row['ans']}\n<SEP>\n")
