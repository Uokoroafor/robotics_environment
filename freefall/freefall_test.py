import random

import arcade

# Want to test objects in freefall of different masses
# and with different initial velocities

# Episode starts with a ball in freefall and stops when it hits the ground (gets to y = 0)

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
GRAVITY = (0, -10)
RADIUS = 20
BOUNCE_FACTOR = 0.9


class Ball:
    def __init__(self, x, y, dx, dy, radius, colour, mass):
        self.x = x
        self.y = y
        self.dx = dx
        self.dy = dy
        self.radius = radius
        self.colour = colour
        self.mass = mass
        # Save info to 2 decimal places
        self.info = f" is at ({round(self.x, 2)}, {round(self.y, 2)}) with velocity ({round(self.dx, 2)}, {round(self.dy, 2)}), " \
                    f"mass {round(self.mass, 2)} and radius {round(self.radius, 2)}."
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

    def print_info(self):
        print(self.info)


    def draw(self):
        arcade.draw_circle_filled(self.x, self.y, self.radius, self.colour)

    # def get_velocity(self):
    #     return math.sqrt(self.dx**2 + self.dy**2)
    #
    # def get_kinetic_energy(self):
    #     return 0.5 * self.mass * self.get_velocity()**2
    #
    # def get_momentum(self):
    #     return self.mass * self.get_velocity()
    #
    # def get_height(self):
    #     return self.y - self.radius
    #
    # def get_potential_energy(self):
    #     return self.mass * GRAVITY[1] * self.get_height()
    #
    # def get_total_energy(self):
    #     return self.get_kinetic_energy() + self.get_potential_energy()

    def check_if_reached_ground(self):
        if self.y - self.radius <= 0:
            return True
        else:
            return False


class Freefall(arcade.Window):
    def __init__(self, width, height):
        super().__init__(width, height, "Freefall")
        self.ball_list = []
        self.saved = False
        self.log = dict(ball_1=None, ball_2=None, answer=None)

    def setup(self):
        arcade.set_background_color(arcade.color.PALE_AQUA)

        # Create two balls with random initial positions and velocities
        for i in range(2):
            x = random.uniform(RADIUS, SCREEN_WIDTH - RADIUS)
            y = random.uniform(RADIUS, SCREEN_HEIGHT - RADIUS)
            dx = random.uniform(-0.1, 0.1)
            dy = random.uniform(-0.1, 0.1)
            # Randomly select a colour
            colour = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            mass = random.uniform(1, 10)
            ball = Ball(x, y, dx, dy, RADIUS, colour, mass)
            self.ball_list.append(ball)
            # self.ball_list[-1].print_info()

            # Add info to log
            self.log[f"ball_{i+1}"] = self.ball_list[-1].info



    def on_draw(self):
        arcade.start_render()
        for ball in self.ball_list:
            ball.draw()

    def update(self, delta_time):
        for ball in self.ball_list:
            ball.update()

            if ball.check_if_reached_ground():
                # print(f"ball {self.ball_list.index(ball)} reached ground - REMOVING")
                # Make this ball the answer if not already filled
                if self.log["answer"] is None:
                    self.log["answer"] = f"ball_{self.ball_list.index(ball)+1}"
                self.ball_list.remove(ball)
                # arcade.close_window()

        if len(self.ball_list) == 0:
            # Append log to file
            with open("freefall_log.txt", "a") as f:
                # Output the log as a string "Ball at (x, y) with velocity (dx, dy),mass m and radius r."
                output = "Two balls are dropped under gravity.\n"
                for key in self.log:
                    if key != "answer":
                        output += f"{key} {self.log[key]}\n"
                    else:
                        output += "Which ball reached the ground first? ball_1, ball_2 or same?\n"
                        output += f"A: {self.log['answer']}\n"

                f.write(output)
            arcade.close_window()

    def on_mouse_press(self, x, y, button, modifiers):
        if not self.saved:
            arcade.save_screen("freefall.png")
            self.saved = True




def main():
    window = Freefall(SCREEN_WIDTH, SCREEN_HEIGHT)
    window.setup()
    arcade.run()


if __name__ == "__main__":
    # Generate 1000 episodes
    for j in range(1000):
        main()
    # main()




