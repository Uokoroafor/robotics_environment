# import arcade
# import random
#
# SCREEN_WIDTH = 800
# SCREEN_HEIGHT = 600
# SCREEN_TITLE = "Coin Pusher"
#
# COIN_RADIUS = 20
# COIN_SPEED_V = 3
# COIN_RATE = 0.1
#
# PUSH_FORCE = 1
# PUSH_RANGE = 60
#
#
# class Coin(arcade.Sprite):
#     def __init__(self, filename, scale, x, y):
#         super().__init__(filename, scale)
#
#         self.center_x = x
#         self.center_y = y
#
#     def update(self):
#         self.center_y -= COIN_SPEED_V
#
#
# class CoinPusher(arcade.Window):
#     def __init__(self, width, height, title):
#         super().__init__(width, height, title)
#
#         arcade.set_background_color(arcade.color.WHITE)
#
#         self.coin_list = arcade.SpriteList()
#
#         self.pusher_x = width // 2
#         self.pusher_y = height // 6
#
#         self.score = 0
#
#     def on_draw(self):
#         arcade.start_render()
#
#         arcade.draw_rectangle_filled(self.pusher_x, self.pusher_y, 100, 50, arcade.color.RED)
#
#         self.coin_list.draw()
#
#         arcade.draw_text(f"Score: {self.score}", 10, 10, arcade.color.BLACK, 18)
#
#     def update(self, delta_time):
#         self.coin_list.update()
#
#         for coin in self.coin_list:
#             if coin.bottom < 0:
#                 coin.remove_from_sprite_lists()
#                 self.score += 1
#
#             if abs(coin.center_x - self.pusher_x) < PUSH_RANGE and coin.top < self.pusher_y:
#                 coin.change_y = PUSH_FORCE
#             else:
#                 coin.change_y = 0
#
#     def on_key_press(self, key, modifiers):
#         if key == arcade.key.SPACE:
#             # x = random.uniform(self.pusher_x - 50, self.pusher_x + 50)
#             # y = self.pusher_y + 50
#             x = random.randint(self.pusher_x - 50, self.pusher_x + 50)
#             y = self.pusher_y + 50
#
#             coin = Coin("../coin.png", 0.5, x, y)
#             self.coin_list.append(coin)
#
#
# def main():
#     game = CoinPusher(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE)
#     arcade.run()
#
#
# if __name__ == "__main__":
#     main()

import arcade
import random
import math

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
SCREEN_TITLE = "Coin Pusher"

COIN_RADIUS = 10
COIN_SPEED_V = 0.5
COIN_SPEED_H = 0.5
COIN_RATE = 0.1

PUSH_FORCE = 1
PUSH_RANGE = 60

SHELF_WIDTH = 100
SHELF_HEIGHT = 20

GRAVITY = (0, -0.1)
BOUNCE_FACTOR = 0.8


class Coin:
    def __init__(self, x, y, radius):

        self.center_x = x
        self.center_y = y
        self.radius = radius
        self.change_x = random.uniform(-COIN_SPEED_H, COIN_SPEED_H)
        self.change_y = -COIN_SPEED_V
        self.bottom = self.center_y - self.radius
        self.top = self.center_y + self.radius

    def update(self):
        self.change_y += GRAVITY[1]
        self.change_x += GRAVITY[0]
        self.center_x += self.change_x
        self.center_y += self.change_y

    def draw(self):
        arcade.draw_circle_filled(self.center_x, self.center_y, self.radius, arcade.color.GOLD)


class Shelf:
    def __init__(self, x, y, width, height, angle=0, color=arcade.color.BROWN):
        self.center_x = x
        self.center_y = y
        self.width = width
        self.height = height
        self.angle = angle
        self.color = color
        self.top = self.center_y + self.height / 2
        self.bottom = self.center_y - self.height / 2

    def draw(self):
        half_width = self.width / 2
        half_height = self.height / 2

        # Define the four corners of the rectangle.
        corners = [
            (-half_width, -half_height),
            (half_width, -half_height),
            (half_width, half_height),
            (-half_width, half_height)
        ]

        # Rotate each corner around the center of the rectangle.
        corners = [(x * math.cos(math.radians(self.angle)) - y * math.sin(math.radians(self.angle)),
                    x * math.sin(math.radians(self.angle)) + y * math.cos(math.radians(self.angle)))
                   for x, y in corners]

        # Offset each corner by the position of the rectangle.
        corners = [(x + self.center_x, y + self.center_y) for x, y in corners]

        arcade.draw_polygon_filled(corners, self.color)


class Pusher:
    def __init__(self, x, y, width, height):
        self.center_x = x
        self.center_y = y
        self.width = width
        self.height = height
        self.color = arcade.color.RED
        self.top = self.center_y + self.height / 2
        self.bottom = self.center_y - self.height / 2

    def draw(self):
        arcade.draw_rectangle_filled(self.center_x, self.center_y, self.width, self.height, self.color)


class CoinPusher(arcade.Window):
    def __init__(self, width, height, title):
        super().__init__(width, height, title)

        arcade.set_background_color(arcade.color.WHITE)

        self.coin_list = []
        self.shelf_list = []

        self.pusher = Pusher(width // 2, height // 6, 100, 50)

        self.score = 0

    def generate_shelves(self):
        for _ in range(5):
            # We want no overlap between shelves
            overlaps = True
            while overlaps:
                x = random.uniform(0, SCREEN_WIDTH - SHELF_WIDTH)
                y = random.uniform(SCREEN_HEIGHT // 2, SCREEN_HEIGHT - SHELF_HEIGHT)
                angle = random.uniform(-45, 45)
                shelf = Shelf(x, y, SHELF_WIDTH, SHELF_HEIGHT, angle)
                overlaps = self.check_shelf_overlaps(shelf)
            self.shelf_list.append(shelf)

    def on_draw(self):
        arcade.start_render()
        self.pusher.draw()

        # arcade.draw_rectangle_filled(self.pusher.center_x, self.pusher.center_y, 100, 50, arcade.color.RED)

        for coin in self.coin_list:
            coin.draw()

        for shelf in self.shelf_list:
            shelf.draw()

        arcade.draw_text(f"Score: {self.score}", 10, 10, arcade.color.BLACK, 18)

    def update(self, delta_time):
        for coin in self.coin_list:
            coin.update()
            coin.draw()


        for coin in self.coin_list:
            if coin.bottom < 0:
                coin.remove_from_sprite_lists()
                self.score += 1

            if abs(coin.center_x - self.pusher.center_x) < PUSH_RANGE and coin.top < self.pusher.center_y:
                coin.change_y = PUSH_FORCE
            # else:
            #     coin.change_y = 0

            for shelf in self.shelf_list:
                if self.check_for_collision(coin, shelf):
                    coin.change_y *= -1 * BOUNCE_FACTOR
                    # update horizontal velocity based on angle of shelf.
                    # If the shelf is flat, the coin will continue in the same direction.
                    # If the shelf is angled, the coin will change direction based on the angle of the shelf.
                    # It adds a horizontal velocity in the direction of the shelf and moves in the direction of the shelf relative to a vertical line.
                    coin.change_x -= math.sin(math.radians(shelf.angle)) * coin.change_y * BOUNCE_FACTOR

            if self.check_for_collision_with_pusher(coin):
                coin.change_y *= -1 * BOUNCE_FACTOR

        # self.shelf_list.update()

    def on_key_press(self, key, modifiers):
        if key == arcade.key.SPACE:
            # Want the coin to be generated at the top of the screen
            x = random.randint(0, SCREEN_WIDTH - self.pusher.width)
            y = SCREEN_HEIGHT

            coin = Coin(x, y, COIN_RADIUS)
            self.coin_list.append(coin)

    def on_mouse_motion(self, x, y, dx, dy):
        self.pusher.center_x = x

    def check_shelf_overlap(self, shelf1, shelf2):
        # Check if shelf1 overlaps with shelf2
        return (shelf1.center_x - shelf1.width / 2 < shelf2.center_x < shelf1.center_x + shelf1.width / 2 and
                shelf1.center_y - shelf1.height / 2 < shelf2.center_y < shelf1.center_y + shelf1.height / 2)

    def check_shelf_overlaps(self, shelf1):
        if len(self.shelf_list) > 1:
            for shelf2 in self.shelf_list:
                if shelf1 != shelf2:
                    if self.check_shelf_overlap(shelf1, shelf2):
                        return True
        return False

    @staticmethod
    def check_for_collision(coin_, shelf_):
        # Check if the bottom of the coin is touching the top of the shelf
        return (shelf_.center_x - shelf_.width / 2 < coin_.center_x < shelf_.center_x + shelf_.width / 2 and
                shelf_.center_y - shelf_.height / 2 < coin_.center_y < shelf_.center_y + shelf_.height / 2)

    def check_for_collision_with_pusher(self, coin_):
        pusher_width = self.pusher.width
        pusher_height = self.pusher.height
        return (self.pusher.center_x - pusher_width / 2 < coin_.center_x < self.pusher.center_x + pusher_width / 2 and
                self.pusher.center_y - pusher_height / 2 < coin_.center_y < self.pusher.center_y + pusher_height / 2)


def main():
    game = CoinPusher(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE)
    game.generate_shelves()
    arcade.run()


if __name__ == "__main__":
    main()
