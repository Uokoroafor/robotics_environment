import arcade

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
SCREEN_TITLE = "Pinball"

BALL_RADIUS = 10
PADDLE_WIDTH = 100
PADDLE_HEIGHT = 20


class Pinball(arcade.Window):
    def __init__(self, width, height, title):
        super().__init__(width, height, title)

        arcade.set_background_color(arcade.color.BLACK)

        self.ball = arcade.SpriteCircle(BALL_RADIUS, arcade.color.RED)
        self.ball.center_x = width // 2
        self.ball.center_y = height // 2
        self.ball.change_x = 3
        self.ball.change_y = 3

        self.paddle = arcade.SpriteSolidColor(
            PADDLE_WIDTH, PADDLE_HEIGHT, arcade.color.BLUE
        )
        self.paddle.center_x = width // 2
        self.paddle.center_y = PADDLE_HEIGHT

    def on_draw(self):
        arcade.start_render()
        self.ball.draw()
        self.paddle.draw()

    def update(self, delta_time):
        self.ball.update()

        if arcade.check_for_collision(self.ball, self.paddle):
            self.ball.change_y *= -1

        if (
            self.ball.center_x < BALL_RADIUS
            or self.ball.center_x > SCREEN_WIDTH - BALL_RADIUS
        ):
            self.ball.change_x *= -1
        if (
            self.ball.center_y < BALL_RADIUS
            or self.ball.center_y > SCREEN_HEIGHT - BALL_RADIUS
        ):
            self.ball.change_y *= -1

        self.ball.center_x += self.ball.change_x
        self.ball.center_y += self.ball.change_y

    def on_key_press(self, key, modifiers):
        if key == arcade.key.LEFT:
            self.paddle.change_x = -5
        elif key == arcade.key.RIGHT:
            self.paddle.change_x = 5

    def on_key_release(self, key, modifiers):
        if key == arcade.key.LEFT or key == arcade.key.RIGHT:
            self.paddle.change_x = 0


def main():
    pinball = Pinball(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE)
    arcade.run()


if __name__ == "__main__":
    main()
