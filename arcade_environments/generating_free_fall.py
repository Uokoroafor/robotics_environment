import random
from typing import Optional

import pandas as pd

# Want to generate a series of free fall actions
GRAVITY = (0, -0.1)


def get_example(num_steps: Optional[int] = 2):
    x_range = (0, 1000)
    y_range = (0, 1000)
    dx_range = (-1, 1)
    dy_range = (-1, 1)
    time_step = 0

    # pick random values for x, y, dx, dy
    x = round(random.uniform(*x_range), 2)
    y = round(random.uniform(*y_range), 2)

    dx = 0  # round(random.uniform(*dx_range), 2)
    dy = round(random.uniform(*dy_range), 2)

    # log initial values
    log_0 = dict(t_0=time_step, x_0=x, y_0=y, dx_0=dx, dy_0=dy)

    for _ in range(num_steps):
        time_step += 1

        # update x, y, dx, dy
        x += dx
        y += dy
        dy += GRAVITY[1]
        dx += GRAVITY[0]
        if y <= 0.0:
            y = 0.0
            dy = 0.0
            break

    # round values
    x = round(x, 2)
    y = round(y, 2)
    dx = round(dx, 2)
    dy = round(dy, 2)

    # add updated values to log
    log_0.update(dict(t_1=time_step, x_1=x, y_1=y, dx_1=dx, dy_1=dy))

    return log_0


def get_example_2balls():
    x1_range = (0, 100)
    y1_range = (0, 100)
    dx_range = (-1, 1)
    dy_range = (0,0)

    # pick random values for x, y, dx, dy
    x1 = round(random.uniform(*x1_range), 2)
    y1 = round(random.uniform(*y1_range), 2)
    dx1 = round(random.uniform(*dx_range), 2)
    dy1 = round(random.uniform(*dy_range), 2)

    x2 = round(random.uniform(*x1_range), 2)
    while x2 == x1:
        x2 = round(random.uniform(*x1_range), 2)
    # make y1 the same as y2 10% of the time
    if random.random() < (1/4):
        y2 = y1
    else:
        y2 = round(random.uniform(*y1_range), 2)

    dx2 = round(random.uniform(*dx_range), 2)
    dy2 = round(random.uniform(*dy_range), 2)

    if y1 == y2:
        ans = 'same'
    elif y1 < y2:
        ans = 'ball_1'
    else:
        ans = 'ball_2'

    # log initial values
    log_0 = dict(x1=x1, y1=y1, dx1=dx1, dy1=dy1, x2=x2, y2=y2, dx2=dx2, dy2=dy2, ans=ans)

    return log_0


def model(x, y, dx, dy, t):
    x_ = x + dx * t
    y_ = max(y + dy * t + 0.5 * GRAVITY[1] * t ** 2, 0)
    dx_ = dx + GRAVITY[0] * t
    dy_ = dy + GRAVITY[1] * t
    if y_ == 0:
        dy_ = 0
    return x_, y_, dx_, dy_


if __name__ == '__main__':
    # Generate 10 examples
    # examples = [get_example() for _ in range(10000)]
    examples = []
    for _ in range(10_000):
        example = get_example_2balls()
        examples.append(example)
    # for _ in range(10000):
    #     example, x_1, y_1, dx_1, dy_1 = get_example()
    #     x, y, dx, dy = example['x_0'], example['y_0'], example['dx_0'], example['dy_0']
    #     t = example['t_1']
    #     x_2, y_2, dx_2, dy_2 = model(x, y, dx, dy, t)
    #     example.update(dict(x_1=x_1, y_1=y_1, dx_1=dx_1, dy_1=dy_1))
    #     examples.append(example)
    #     assert round(y_1,2) == round(y_2,2), f"y_1: {y_1}, y_2: {y_2} should be equal"

    df = pd.DataFrame(examples)
    # remove duplicates
    df.drop_duplicates(inplace=True)
    # print number of examples
    print(f"Number of examples: {len(df)}")
    # save to csv
    df.to_csv('ball_drop.csv', index=False)

    print(df.head())
