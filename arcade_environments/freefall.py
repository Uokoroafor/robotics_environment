import arcade
import pymunk
from typing import Tuple, List, Any
from arcade_environments.arcade_objects import Ball
import random
import time


class FreefallView(arcade.View):
    def __init__(self, env_: Any):
        """ This is a view for the freefall environment. It is a wrapper around arcade's view object.

        Args:
            env_ (Any): The environment object
        """

        super().__init__()
        self.env = env_

    def on_draw(self):
        arcade.start_render()
        for obj in self.env.objects:
            obj.draw()

    def on_update(self, delta_time):
        self.env.step()


class Freefall:

    def __init__(self, width: int, height: int, gravity_x: float = 0.0, gravity_y: float = -0.1, radius: float = 20,
                 height_limit: float = 1, render: bool = True):
        """ This is a freefall environment. It is a wrapper around pymunk's space object. It is a simple environment
        where a ball falls from the top of the screen to the bottom. There is no bounce or collision detection.

        Args:
            width (int): The width of the environment
            height (int): The height of the environment
            gravity_x (float, optional): The x component of the gravity vector. Defaults to 0.0.
            gravity_y (float, optional): The y component of the gravity vector. Defaults to -1.0.
            radius (float, optional): The radius of the ball. Defaults to 20.
            height_limit (float, optional): The height limit of the generated ball. Defaults to 1.
            render (bool, optional): Whether to render the environment. Defaults to True.
        """

        self.width = width
        self.height = height
        self.gravity = (gravity_x, gravity_y)
        self.radius = radius
        self.space = pymunk.Space()
        self.space.gravity = self.gravity
        self.height_limit = height_limit
        self.render = render
        self.objects = []
        self.text_log = []
        self.numerical_log = dict(
            gravity=self.gravity[-1],
            radius=self.radius,
            x_0=None,
            y_0=None,
            vy_0=None,
            t_1=None,
            x_1=None,
            y_1=None,
            vy_1=None,
        )
        # Add gravity to the self.text_log
        self.text_log.append(f"Gravity: {self.gravity[-1]}.")

        self.add_ball()
        if self.render:
            self.setup_render()

    def add_ball(self):
        """ This function adds a ball to the environment. The ball is added to the self.objects list and the self.space object.
        """

        # Pick a random x and y coordinate for the ball
        x = round(random.uniform(0, self.width), 2)
        y = round(random.uniform((1 - self.height_limit) * self.height, self.height), 2)

        # Create a ball object
        ball = Ball(x, y, self.radius)

        # Add the ball to the self.objects list and the self.space object
        self.objects.append(ball)
        self.space.add(ball.body, ball.shape)
        # Add the ball to the self.text_log
        self.text_log.append(f"Added ball of radius {self.radius} at x={x} and y={y}).")
        self.text_log.append(f"Initial velocity is 0.")

        # Add the ball's initial position to the self.numerical_log
        self.numerical_log["x_0"] = x
        self.numerical_log["y_0"] = y

        # Initial velocity is 0
        self.numerical_log["vy_0"] = 0.0

    def drop_ball(self, time: float = 5.0):
        """This function drops the ball for a given amount of time.

        Args:
            time (float, optional): The amount of time to drop the ball for. Defaults to 5.0.

        Returns:
            List[float]: The state of the environment after the ball has been dropped.
        """
        # Drop the ball for a given amount of time (in seconds)
        dt = 1 / 60  # Fixed time step of 1/60 seconds
        num_steps = int(time / dt)

        for _ in range(num_steps):
            self.step()

        # Add the ball's final position to the self.numerical_log
        self.numerical_log["t_1"] = time
        self.numerical_log["x_1"] = round(self.objects[0].body.position.x, 2)
        self.numerical_log["y_1"] = round(self.objects[0].body.position.y, 2)
        self.numerical_log["vy_1"] = round(self.objects[0].body.velocity.y, 2)

        # Add the ball's final position to the self.text_log
        self.text_log.append(
            f"Ball's position at time {time} is x={round(self.objects[0].body.position.x, 2)} and y={round(self.objects[0].body.position.y, 2)}.")
        self.text_log.append(f"Ball's final velocity is {round(self.objects[0].body.velocity.y, 2)}.")

        # Stop the rendering and close the window
        if self.render:
            arcade.finish_render()
            arcade.close_window()

        # Return the state of the environment

        return self.get_state()

    def step(self):
        """ This function steps the environment forward one step.
        """
        self.space.step(1 / 60)
        if self.render:
            self.draw()

    def draw(self):
        """ This function draws the environment.
        """
        arcade.start_render()  # Start rendering
        for obj in self.objects:
            obj.draw()
        arcade.finish_render()  # Finish rendering

    def get_state(self) -> List[float]:
        """ This function returns the state of the environment. The state is a list of the x and y
        coordinates of the ball.

        Returns:
            List[float]: The state of the environment
        """
        return [self.objects[0].body.position.x, self.objects[0].body.position.y, self.objects[0].body.velocity.y]

    def setup_render(self):
        """ This function sets up the rendering of the environment.
        """
        arcade.open_window(self.width, self.height, "Freefall")
        arcade.set_background_color(arcade.color.WHITE)
        self.draw()

    def close_window(self):
        arcade.close_window()

    def render_env(self):
        """ Start the arcade loop for rendering """
        window = arcade.Window(self.width, self.height, "Freefall")
        arcade.set_background_color(arcade.color.WHITE)
        view = FreefallView(self)
        window.show_view(view)
        arcade.run()


if __name__ == "__main__":
    print('Creating environment')
    env = Freefall(500, 500, render=True, height_limit=0.2)
    if env.render:
        env.render_env()

    print('Dropping ball')
    env.drop_ball()
    time.sleep(2)  # Delay for 2 seconds
    env.close_window()

    print(env.numerical_log)
    print(env.text_log)
