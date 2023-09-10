# Read in files
import pandas as pd

# File is examples_same_steps.csv
df_same = pd.read_csv("../data/freefall/examples_same_steps.csv")
df_diff = pd.read_csv("../data/freefall/examples_diff_steps.csv")
df_tests = pd.read_csv("../data/freefall/examples_tests.csv")

# Want to convert the dataframes to lines of text
# An object is dropped at time {t_0} with initial position {x_0}, {y_0} and initial velocity {dx_0}, {dy_0}.
# Gravity is {GRAVITY[1]}. What is the position of the vertical object at time {t_1}?  Answer: {y_1}

GRAVITY = (0, -0.1)
# Generate text for same steps
with open("../data/freefall/examples_same_steps.txt", "w") as f:
    for _, row in df_same.iterrows():
        f.write(
            f"An object is dropped at time {row['t_0']} with initial position {row['x_0']}, {row['y_0']} and initial velocity {row['dx_0']}, {row['dy_0']}.\n"
        )
        f.write(
            f"Acceleration under Gravity is {GRAVITY[1]}. What is the position of the vertical object at time {row['t_1']}?  Answer: {row['y_1']}\n<SEP>\n"
        )

# Generate text for different steps
with open("../data/freefall/examples_diff_steps.txt", "w") as f:
    for _, row in df_diff.iterrows():
        f.write(
            f"An object is dropped at time {row['t_0']} with initial position {row['x_0']}, {row['y_0']} and initial velocity {row['dx_0']}, {row['dy_0']}.\n"
        )
        f.write(
            f"Acceleration under Gravity is {GRAVITY[1]}. What is the position of the vertical object at time {row['t_1']}?  Answer: {row['y_1']}\n<SEP>\n"
        )

# Generate text for tests
with open("../data/freefall/examples_tests.txt", "w") as f:
    for _, row in df_tests.iterrows():
        f.write(
            f"An object is dropped at time {row['t_0']} with initial position {row['x_0']}, {row['y_0']} and initial velocity {row['dx_0']}, {row['dy_0']}.\n"
        )
        f.write(
            f"Acceleration under Gravity is {GRAVITY[1]}. What is the position of the vertical object at time {row['t_1']}?  Answer: {row['y_1']}\n<SEP>\n"
        )
