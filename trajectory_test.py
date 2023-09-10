import arcade
import math
import random

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
GRAVITY = (0, -0.1)
RADIUS = 20

# Here two objects are projected from the same height with the same initial velocity but different masses
# The episode ends when both objects hit the ground (get to y = 0)


class Ball:
    def __init__(self, x, y, dx, dy, radius, colour, mass):
        self.x = x
        self.y = y
        self.dx = dx
        self.dy = dy
        self.radius = radius
        self.colour = colour
        self.mass = mass

    def update(self):
        self.x += self.dx
        self.y += self.dy
        self.dy += GRAVITY[1]

        if self.y - self.radius < 0:
            self.y = self.radius
            self.dy *= -1
        elif self.y + self.radius > SCREEN_HEIGHT:
            self.y = SCREEN_HEIGHT - self.radius
            self.dy *= -1

    def print_info(self):
        print(
            f"Ball at ({self.x}, {self.y}) with velocity ({self.dx}, {self.dy}), "
            f"mass {self.mass} and radius {self.radius}"
        )

    def draw(self):
        arcade.draw_circle_filled(self.x, self.y, self.radius, self.colour)

    def get_velocity(self):
        return math.sqrt(self.dx**2 + self.dy**2)

    def get_kinetic_energy(self):
        return 0.5 * self.mass * self.get_velocity() ** 2

    def get_momentum(self):
        return self.mass * self.get_velocity()

    def get_height(self):
        return self.y - self.radius

    def get_potential_energy(self):
        return self.mass * GRAVITY[1] * self.get_height()
