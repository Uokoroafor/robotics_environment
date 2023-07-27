# Testing accelaration due to gravity

import arcade
import csv
import random

# Want to test objects in freefall of different masses
# and with different initial velocities

# Episode starts with a ball in freefall and stops when it hits the ground (gets to y = 0)

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
GRAVITY = (0, -0.1)
RADIUS = 0
BOUNCE_FACTOR = 0.9


class Ball:
    def __init__(self, x, y, dx, dy, radius, colour, mass, time_step=0):
        self.x = x
        self.y = y
        self.dx = dx
        self.dy = dy
        self.radius = radius
        self.colour = colour
        self.mass = mass
        self.time_step = time_step
        # Save info to 2 decimal places
        self.info = f"At time {self.time_step} the ball is at ({round(self.x, 2)}, {round(self.y, 2)}) with velocity " \
                    f"({round(self.dx, 2)}, {round(self.dy, 2)}) and radius {round(self.radius, 2)}.\n"
        self.pos_dict = dict(timestep=self.time_step, pos_x=round(self.x, 2), pos_y=round(self.y, 2),
                             vel_x=round(self.dx, 2),
                             vel_y=round(self.dy, 2))

    def update(self):
        self.x += self.dx
        self.y += self.dy
        self.dy += GRAVITY[1]

        if self.y - self.radius < 0:
            self.y = self.radius
            self.dy *= -BOUNCE_FACTOR
        elif self.y + self.radius > SCREEN_HEIGHT:
            self.y = SCREEN_HEIGHT - self.radius
            self.dy *= -BOUNCE_FACTOR

        self.update_info()

    def print_info(self):
        print(self.info)

    def draw(self):
        arcade.draw_circle_filled(self.x, self.y, self.radius, self.colour)

    def check_if_reached_ground(self):
        if self.y - self.radius <= 0:
            return True
        else:
            return False

    def update_info(self):
        self.info = f"At time {self.time_step} the ball is at ({round(self.x, 2)}, {round(self.y, 2)}) with velocity " \
                    f"({round(self.dx, 2)}, {round(self.dy, 2)}) and radius {round(self.radius, 2)}.\n <SEP> \n"
        self.pos_dict = dict(timestep=self.time_step, pos_x=round(self.x, 2), pos_y=round(self.y, 2),
                             vel_x=round(self.dx, 2), vel_y=round(self.dy, 2))


class Freefall(arcade.Window):
    def __init__(self, width, height, j):
        super().__init__(width, height, "Freefall")
        self.time_step = 0
        self.ball_list = []
        self.saved = False
        self.log = dict(episode=dict(episode=j), input=None, output=None)
        self.info = ''

    def setup(self):
        arcade.set_background_color(arcade.color.PALE_AQUA)

        # Create a ball with random initial positions and velocities
        x = random.uniform(RADIUS, SCREEN_WIDTH - RADIUS)
        y = random.uniform(RADIUS, SCREEN_HEIGHT - RADIUS)
        dx = random.uniform(-0.1, 0.1)
        dy = random.uniform(-0.1, 0.1)

        # Randomly select a colour
        colour = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        mass = random.uniform(1, 10)
        ball = Ball(x, y, dx, dy, RADIUS, colour, mass)
        self.ball_list.append(ball)
        self.time_step = 0
        self.log["input"] = ball.pos_dict
        self.info += ball.info

    def on_draw(self):
        arcade.start_render()
        for ball in self.ball_list:
            ball.draw()

    def update(self, delta_time):
        for ball in self.ball_list:
            ball.update()

            if ball.check_if_reached_ground():
                assert ball.dy <= 0

        self.time_step += 1
        ball.time_step = self.time_step
        if self.time_step == 6:
            self.log["output"] = ball.pos_dict
            self.info += ball.info
            # write log to a csv file
            with open("freefall_log.csv", "a") as f:
                writer = csv.DictWriter(f, fieldnames=["episode", "input", "output"])
                writer.writerow(self.log)

            # write log to a txt file
            with open("freefall_log_.txt", "a") as f:
                f.write(self.info)

            arcade.close_window()

    def on_mouse_press(self, x, y, button, modifiers):
        print('Mouse button pressed')
        if not self.saved:
            # Save an image of the window
            arcade.save_screenshot(f"freefall.png")
            self.saved = True


def main(j):
    window = Freefall(SCREEN_WIDTH, SCREEN_HEIGHT, j)
    window.setup()
    arcade.run()


if __name__ == "__main__":
    # Generate 1000 episodes
    for j in range(1000):
        main(j)
