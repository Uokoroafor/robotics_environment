import arcade
import random

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
BALL_RADIUS = 20
GRAVITY = 0.1
BOUNCE_FACTOR = 0.9

class Ball:
    def __init__(self, x, y, dx, dy, radius, color):
        self.x = x
        self.y = y
        self.dx = dx
        self.dy = dy
        self.radius = radius
        self.color = color

    def update(self):
        self.x += self.dx
        self.y += self.dy
        self.dy -= GRAVITY

        if self.x - self.radius < 0:
            self.x = self.radius
            self.dx *= -BOUNCE_FACTOR
        elif self.x + self.radius > SCREEN_WIDTH:
            self.x = SCREEN_WIDTH - self.radius
            self.dx *= -BOUNCE_FACTOR

        if self.y - self.radius < 0:
            self.y = self.radius
            self.dy *= -BOUNCE_FACTOR
        elif self.y + self.radius > SCREEN_HEIGHT:
            self.y = SCREEN_HEIGHT - self.radius
            self.dy *= -BOUNCE_FACTOR

    def draw(self):
        arcade.draw_circle_filled(self.x, self.y, self.radius, self.color)


class BouncingBalls(arcade.Window):
    def __init__(self, width, height):
        super().__init__(width, height, "Bouncing Balls")
        self.ball_list = []
        self.saved = False

    def setup(self):
        # Light pink background
        arcade.set_background_color(arcade.color.PALE_AQUA)

        # Create two balls with random initial positions and velocities
        for _ in range(5):
            x = random.uniform(BALL_RADIUS, SCREEN_WIDTH - BALL_RADIUS)
            y = random.uniform(BALL_RADIUS, SCREEN_HEIGHT - BALL_RADIUS)
            dx = random.uniform(-5, 5)
            dy = random.uniform(-5, 5)
            # Randomly select a color
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            ball = Ball(x, y, dx, dy, BALL_RADIUS, color)
            self.ball_list.append(ball)

    def on_draw(self):
        arcade.start_render()
        for ball in self.ball_list:
            ball.draw()
        if not self.saved:
            # arcade.finish_render()
            arcade.get_image().save('../test3.png')
            self.saved = True

    def update(self, delta_time):
        for ball in self.ball_list:
            ball.update()
        self.check_for_collisions()

    # def on_key_press(self, key, modifiers):
    #     if key == arcade.key.ESCAPE:
    #         arcade.close_window()

    def check_for_collisions(self):
        for i in range(len(self.ball_list)):
            for j in range(i + 1, len(self.ball_list)):
                ball1 = self.ball_list[i]
                ball2 = self.ball_list[j]

                distance = ((ball1.x - ball2.x) ** 2 + (ball1.y - ball2.y) ** 2) ** 0.5
                if distance < ball1.radius + ball2.radius:
                    # Collision detected, reverse velocities
                    ball1.dx *= -1
                    ball1.dy *= -1
                    ball2.dx *= -1
                    ball2.dy *= -1


def main():
    game = BouncingBalls(SCREEN_WIDTH, SCREEN_HEIGHT)
    game.setup()
    arcade.run()


if __name__ == "__main__":
    main()
