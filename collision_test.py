import arcade
import random
import math

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
BALL_RADIUS = 20


# Want to test collisions with a ball and a line segment


class BouncingBalls(arcade.Window):
    def __init__(self, width, height):
        super().__init__(width, height, "Bouncing Balls")
        self.ball_list = []
        self.line_list = []
        self.saved = False
        self.frames = []

    def setup(self):
        # Light pink background
        arcade.set_background_color(arcade.color.PALE_AQUA)

        # Create two balls with random initial positions and velocities
        for _ in range(2):
            x = random.uniform(BALL_RADIUS, SCREEN_WIDTH - BALL_RADIUS)
            y = random.uniform(BALL_RADIUS, SCREEN_HEIGHT - BALL_RADIUS)
            dx = random.uniform(-5, 5)
            dy = random.uniform(-5, 5)
            # Randomly select a color
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            ball = Ball(x, y, dx, dy, BALL_RADIUS, color)
            self.ball_list.append(ball)

        # Create a line segment
        # self.line_list.append(Line(100, 100, 200, 200, arcade.color.BLACK))
        # self.line_list.append(Line(100, 200, 200, 100, arcade.color.BLACK))
        # self.line_list.append(Line(100, 100, 100, 300, arcade.color.BLACK))
        # self.line_list.append(Line(100, 100, 400, 100, arcade.color.BLACK))

    def on_draw(self):
        arcade.start_render()
        for ball in self.ball_list:
            ball.draw()
        for line in self.line_list:
            line.draw()
        # image = arcade.get_image()


    def update(self, delta_time):
        for ball in self.ball_list:
            ball.update()

        # Check for collisions between balls
        for ball1 in self.ball_list:
            for ball2 in self.ball_list:
                if ball1 != ball2:
                    if ball1.check_for_collision(ball2):
                        ball1.resolve_collision(ball2)

        # Check for collisions between balls and lines
        for ball in self.ball_list:
            for line in self.line_list:
                if ball.check_for_collision(line):
                    ball.resolve_collision(line)

    # def on_mouse_press(self, x, y, button, modifiers):
    #     # Save the balls to a file
    #     if not self.saved:
    #         self.save_balls()
    #         self.saved = True

    def save_balls(self):
        # Save the balls to a file
        with open("balls.txt", "w") as f:
            for ball in self.ball_list:
                f.write(
                    f"{ball.x} {ball.y} {ball.dx} {ball.dy} {ball.radius} {ball.color[0]} {ball.color[1]} {ball.color[2]}\n")

    def load_balls(self):
        # Load the balls from a file
        with open("balls.txt", "r") as f:
            for line in f:
                x, y, dx, dy, radius, r, g, b = line.split()
                x = float(x)
                y = float(y)
                dx = float(dx)
                dy = float(dy)
                radius = float(radius)
                r = int(r)
                g = int(g)
                b = int(b)
                color = (r, g, b)
                ball = Ball(x, y, dx, dy, radius, color)
                self.ball_list.append(ball)

        self.saved = True

    # def on_key_press(self, key, modifiers):
    #     # Load the balls from a file
    #     self.load_balls()

    # Save rendering as gif
    def on_close(self):
        self.save_balls()
        super().on_close()


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

        if self.x - self.radius < 0:
            self.x = self.radius
            self.dx *= -1
        elif self.x + self.radius > SCREEN_WIDTH:
            self.x = SCREEN_WIDTH - self.radius
            self.dx *= -1

        if self.y - self.radius < 0:
            self.y = self.radius
            self.dy *= -1
        elif self.y + self.radius > SCREEN_HEIGHT:
            self.y = SCREEN_HEIGHT - self.radius
            self.dy *= -1

    def draw(self):
        arcade.draw_circle_filled(self.x, self.y, self.radius, self.color)

    def check_for_collision(self, obj):
        if isinstance(obj, Ball):
            return self.check_for_collision_ball(obj)
        elif isinstance(obj, Line):
            return self.check_for_collision_line(obj)

    def check_for_collision_ball(self, ball):
        distance = ((self.x - ball.x) ** 2 + (self.y - ball.y) ** 2) ** 0.5
        if distance <= self.radius + ball.radius:
            return True

        else:
            return False

    def check_for_collision_line(self, line):
        # If the shortest distance from the center of the ball to the line is less than the radius of the ball, then
        # there is a collision

        # Calculate the shortest distance from the center of the ball to the line
        A = line.y2 - line.y1
        B = line.x1 - line.x2
        C = (line.x2 * line.y1) - (line.y2 * line.x1)
        distance = abs(A * self.x + B * self.y + C) / ((A ** 2 + B ** 2) ** 0.5)

        if distance <= self.radius:
            # print("Collision detected between a ball and a line")
            # print(f"Distance: {distance} vs {self.radius}")
            # print(f"Ball: {self.x}, {self.y}")
            # print(f"Line: {line.x1}, {line.y1}, {line.x2}, {line.y2}")
            # print(f"A: {A}, B: {B}, C: {C}")

            return True
        else:
            return False

    def resolve_collision(self, obj):
        if isinstance(obj, Ball):
            self.resolve_collision_ball(obj)
        elif isinstance(obj, Line):
            self.resolve_collision_line(obj)

    def resolve_collision_ball(self, ball):
        # Calculate the angle between the two balls
        angle = math.atan2(ball.y - self.y, ball.x - self.x)

        # Calculate the angle of the ball
        ball_angle = math.atan2(self.dy, self.dx)

        # Calculate the angle of reflection
        reflection_angle = 2 * angle - ball_angle

        # Calculate the new velocity
        velocity = (self.dx ** 2 + self.dy ** 2) ** 0.5

        # Calculate the new velocity components
        self.dx = velocity * math.cos(reflection_angle)
        self.dy = velocity * math.sin(reflection_angle)

    def resolve_collision_line(self, line):
        # Calculate the shortest distance from the center of the ball to the line
        # A = line.y2 - line.y1
        # B = line.x1 - line.x2
        # C = (line.x2 * line.y1) - (line.y2 * line.x1)
        # distance = abs(A * self.x + B * self.y + C) / ((A ** 2 + B ** 2) ** 0.5)

        # Calculate the angle of the line
        angle = math.atan2(line.y2 - line.y1, line.x2 - line.x1)

        # Calculate the angle of the ball
        ball_angle = math.atan2(self.dy, self.dx)

        # Calculate the angle of reflection
        reflection_angle = 2 * angle - ball_angle

        # Calculate the new velocity
        velocity = (self.dx ** 2 + self.dy ** 2) ** 0.5

        # Calculate the new velocity components
        self.dx = velocity * math.cos(reflection_angle)
        self.dy = velocity * math.sin(reflection_angle)

        # # Move the ball to the edge of the line
        # self.x = line.x1 + (self.radius + distance) * math.cos(angle)
        # self.y = line.y1 + (self.radius + distance) * math.sin(angle)


class Line:
    def __init__(self, x1, y1, x2, y2, color):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.color = color

    def draw(self):
        arcade.draw_line(self.x1, self.y1, self.x2, self.y2, self.color, 3)


def main():
    window = BouncingBalls(SCREEN_WIDTH, SCREEN_HEIGHT)
    window.setup()
    arcade.run()



if __name__ == "__main__":
    main()
