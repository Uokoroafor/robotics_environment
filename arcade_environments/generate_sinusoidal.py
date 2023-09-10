import random
from typing import Optional, List, Tuple, Callable
import math

import pandas as pd
import random

# Want to generate a series of sinusoidal functions
train_max = 10
train_min = 0
test_max = 15
test_min = 10

func1 = lambda x: math.sin(x)
func2 = lambda x: math.cos(x)
func3 = lambda x: math.sin(x) + math.cos(x)
func4 = lambda x: x * math.sin(x)
func5 = lambda x: x * math.cos(x)
func6 = lambda x: x * math.sin(x) + x * math.cos(x)

# Set the seed
random.seed(6_345_789)
# Wilson Pickett - 634-5789 https://www.youtube.com/watch?v=TSGuaVAufV0


def get_examples(num: int, data_type: str, func: Callable) -> List[Tuple[float, float]]:
    """Generate a random example of a sin function
    Args:
        num: number of steps to generate
        data_type: train or test
        func: the function to use
    Returns:
        list of tuples of (x, y) values
    """
    output = []
    if data_type == "train":
        _min = train_min
        _max = train_max
    elif data_type == "test":
        _min = test_min
        _max = test_max
    for _ in range(num):
        x = random.uniform(_min, _max) * random.choice([-1, 1])
        y = func(x)
        output.append((round(x, 2), round(y, 4)))
    return output


func_list = [func1, func2, func3, func4, func5, func6]
func_names = ["sin", "cos", "sin+cos", "xsinx", "xcosx", "xsinx+xcosx"]

# generate 10000 examples of each function and save to csv
num_examples_train = 10_000
num_examples_test = 1_000

# test_examples
#
# print('Generating train data')
# train_data = []
#
# train_data += get_examples(num_examples_train, 'train', func1)
# # train_data += get_cos_example(100, 'train')
# print(train_data)

minimal_text_template = "x = {x} Ans: y = {y}"
descriptive_text_template = "The value of x is {x}. What is the value of y? Ans: {y}"
function_include_text_template = (
    "y = {func_name}(x). The value of x is {x}. What is the value of y? Ans: {y}"
)

folder_name = "../sinusoidal_functions/"
# For each function, generate csv, minimal text, descriptive text, function include text
for func, func_name in zip(func_list, func_names):
    print("Generating data for function {}".format(func_name))
    train_data = get_examples(num_examples_train, "train", func)
    train_df = pd.DataFrame(train_data, columns=["x", "y"])
    train_df.to_csv(folder_name + "train_" + func_name + ".csv", index=False)

    test_data = get_examples(num_examples_test, "test", func)
    test_df = pd.DataFrame(test_data, columns=["x", "y"])
    test_df.to_csv(folder_name + "test_" + func_name + ".csv", index=False)

    # Generate minimal text
    minimal_text = []
    for x, y in train_data:
        minimal_text.append(minimal_text_template.format(x=x, y=y))
    with open(folder_name + "minimal_text_" + func_name + ".txt", "w") as f:
        f.write("\n".join(minimal_text))

    # Generate descriptive text
    descriptive_text = []
    for x, y in train_data:
        descriptive_text.append(descriptive_text_template.format(x=x, y=y))
    with open(folder_name + "descriptive_text_" + func_name + ".txt", "w") as f:
        f.write("\n".join(descriptive_text))

    # Generate function include text
    function_include_text = []
    for x, y in train_data:
        function_include_text.append(
            function_include_text_template.format(func_name=func_name, x=x, y=y)
        )
    with open(folder_name + "function_include_text_" + func_name + ".txt", "w") as f:
        f.write("\n".join(function_include_text))
