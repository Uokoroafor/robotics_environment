import csv
import random
from typing import Optional

import arcade
import pymunk

from arcade_environments.arcade_objects import Ball, Ground
from arcade_environments.render_constants import (
    SCREEN_WIDTH as WIDTH,
    SCREEN_HEIGHT as HEIGHT,
)

# Constants
BALL_RADIUS = 10
GRAVITY = -10
DAMPING = 1.0


class FreefallView(arcade.View):
    def __init__(self, env):
        super().__init__()
        self.env = env

    def on_draw(self):
        arcade.start_render()
        for obj in self.env.objects:
            obj.draw()

    def on_update(self, delta_time):
        self.env.update(delta_time)


class FreefallEnv:
    def __init__(self, render=True, height_limit=0.2, time_limit=5):
        self.space = pymunk.Space()
        self.space.gravity = (0, GRAVITY)
        self.space.damping = DAMPING
        self.objects = []
        self.text_log = []
        self.text_log.append(f"Gravity is {GRAVITY}.")
        self.numerical_log = dict(
            gravity=GRAVITY,
            radius=BALL_RADIUS,
            y_0=None,
            vy_0=None,
            t_1=None,
            y_1=None,
            vy_1=None,
        )
        self.ball = None
        self.width = WIDTH
        self.height = HEIGHT
        self.height_limit = height_limit
        self.time_limit = time_limit
        self.elapsed_time = 0.0

        # Add ground
        ground = Ground(self.space, width=WIDTH)
        ground.elasticity = 0.0  # Make the ground not bouncy

        self.add_ball()
        self.render_mode = render
        self.end_episode = False

        if self.render_mode:
            self.window = arcade.Window(WIDTH, HEIGHT, "Freefall")
            self.window.show_view(FreefallView(self))

        else:
            while True:  # Infinite loop for logging mode
                self.update(1 / 60)  # Update every 1/60 seconds
                if self.end_episode:
                    break

    def update(self, delta_time):
        # if self.render_mode:
        self.space.step(delta_time)  # Advance the simulation by 1/60th of a second
        self.elapsed_time += delta_time
        # else:
        #     self.space.step(self.time_limit)
        #     self.elapsed_time += self.time_limit

        # End the episode if the ball hits the ground
        if self.elapsed_time >= self.time_limit:
            self.log_state(self.elapsed_time)
            self.end_episode = True

        if self.end_episode and self.render_mode:
            arcade.close_window()

    def add_ball(self):
        """Makes a ball and adds it to the environment"""
        # Pick a random x and y coordinate for the ball
        x = round(random.uniform(0, self.width), 2)
        y = round(random.uniform((1 - self.height_limit) * self.height, self.height), 2)
        self.ball = Ball(self.space, radius=BALL_RADIUS, mass=1, x=x, y=y)
        self.objects.append(self.ball)
        # Update the initial conditions in the numerical log and text log
        self.numerical_log["y_0"] = round(y, 2)
        self.numerical_log["vy_0"] = round(self.ball.body.velocity.y, 2)
        self.text_log.append(
            f"Ball of radius {BALL_RADIUS} dropped from y={y:.2f} with initial velocity={float(0)}."
        )

    def log_state(self, t):
        self.numerical_log["t_1"] = round(self.elapsed_time, 1)
        self.numerical_log["y_1"] = round(self.ball.body.position.y, 2)
        self.numerical_log["vy_1"] = round(self.ball.body.velocity.y, 2)
        self.text_log.append(f"At time {round(t, 1)}")
        self.text_log.append(
            f"Ball's Position is y={round(self.ball.body.position.y, 2)}."
        )
        self.text_log.append(f"Velocity is {round(self.ball.body.velocity.y, 2)}.")

    def make_minimal_text(
        self, split_text: str = "ans: ", ans_key: Optional[str] = None
    ) -> str:
        """Joins up the text in the numerical log to make a minimal text"""
        # Join the column names and values
        if ans_key is None:
            # Take the last value in the numerical log as the answer
            ans_key = list(self.numerical_log.keys())[-1]
        text = ""
        for key, value in self.numerical_log.items():
            if key != ans_key:
                text += f"{key}: {value}, "

        # Add the split text and the answer
        text += f"{split_text}{self.numerical_log[ans_key]}"
        return text

    def make_descriptive_text(
        self, split_text: str = "ans: ", ans_index: Optional[int] = None
    ) -> str:
        """Joins up the text in the text log to make a descriptive text"""
        # Join the column names and values
        if ans_index is None:
            # Take the last value in the numerical log as the answer
            ans_index = len(self.text_log) - 1
        text = ""
        for i, line in enumerate(self.text_log):
            if i != ans_index:
                text += f"{line} "

        # Add the split text and the answer
        text += f"{split_text}{self.text_log[ans_index]}"
        return text


def main(iters=1000, save_path=None):
    numerical_logs = []
    text_logs = []
    minimal_texts = []
    descriptive_texts = []

    for i in range(iters):
        time_limit = random.randint(1, 10)
        env = FreefallEnv(render=False, time_limit=time_limit)
        numerical_logs.append(env.numerical_log)
        text_logs.append(env.text_log)
        minimal_texts.append(env.make_minimal_text())
        descriptive_texts.append(env.make_descriptive_text())

    # Remove duplicates
    numerical_logs = list({tuple(log.items()) for log in numerical_logs})
    text_logs = list({tuple(log) for log in text_logs})
    minimal_texts = list({text for text in minimal_texts})
    descriptive_texts = list({text for text in descriptive_texts})

    # Reconvert numerical logs to dicts
    numerical_logs = [dict(log) for log in numerical_logs]
    text_logs = [" ".join(list(log)) for log in text_logs]

    if save_path is not None:
        # save numerical log as csv and the rest as txt
        # All the fields in the numerical log are the same for each entry so save it as a csv
        with open(save_path + "numerical_logs.csv", "w") as f:
            writer = csv.DictWriter(f, fieldnames=numerical_logs[0].keys())
            writer.writeheader()
            for log in numerical_logs:
                writer.writerow(log)

        with open(save_path + "text_log.txt", "w") as f:
            for log in text_logs:
                f.write(str(log) + "\n")
        with open(save_path + "minimal_text.txt", "w") as f:
            for text in minimal_texts:
                f.write(str(text) + "\n")
        with open(save_path + "descriptive_text.txt", "w") as f:
            for text in descriptive_texts:
                f.write(str(text) + "\n")

    print(f"Number of unique numerical logs: {len(numerical_logs)}")
    print(f"Number of unique text logs: {len(text_logs)}")
    print(f"Number of unique minimal texts: {len(minimal_texts)}")
    print(f"Number of unique descriptive texts: {len(descriptive_texts)}")

    return numerical_logs, text_logs, minimal_texts, descriptive_texts


if __name__ == "__main__":
    random.seed(6_345_789)
    save_path_ = "../data/freefall/freefall_"
    main(100_000, save_path_)
