import arcade

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
SCREEN_TITLE = "Table Arrangement"

TABLE_WIDTH = 500
TABLE_HEIGHT = 300
TABLE_X = (SCREEN_WIDTH - TABLE_WIDTH) // 2
TABLE_Y = (SCREEN_HEIGHT - TABLE_HEIGHT) // 2

ROBOT_SIZE = 50
OBJECT_SIZE = 30


class Robot(arcade.Sprite):
    def __init__(self, x, y):
        super().__init__()

        self.center_x = x
        self.center_y = y

    def perform_action(self, objects):
        for obj in objects:
            if self.intersects(obj):
                # Perform some action based on object type
                if obj.object_type == "apple":
                    obj.change_x = 2
                elif obj.object_type == "banana":
                    obj.change_y = -2

    def intersects(self, obj):
        return (
            abs(self.center_x - obj.center_x) < (ROBOT_SIZE + OBJECT_SIZE) / 2
            and abs(self.center_y - obj.center_y) < (ROBOT_SIZE + OBJECT_SIZE) / 2
        )


class Object:
    def __init__(self, x, y, object_type):
        self.center_x = x
        self.center_y = y
        self.change_x = 0
        self.change_y = 0
        self.object_type = object_type
        self.area = None

    def update(self, objects):
        self.center_x += self.change_x
        self.center_y += self.change_y

        # Check for collision with other objects
        for obj in objects:
            if obj is not self and self.intersects(obj):
                self.change_x = 0
                self.change_y = 0
                break

        # Check if the object is in its area
        if self.area is not None and not self.intersects(self.area):
            self.change_x = 0
            self.change_y = 0

    def intersects(self, obj):
        return (
            abs(self.center_x - obj.center_x) < OBJECT_SIZE
            and abs(self.center_y - obj.center_y) < OBJECT_SIZE
        )

    def draw(self):
        if self.object_type == "apple":
            arcade.draw_circle_filled(
                self.center_x, self.center_y, OBJECT_SIZE, arcade.color.RED
            )
            arcade.draw_circle_outline(
                self.area.center_x, self.area.center_y, OBJECT_SIZE, arcade.color.RED
            )
        elif self.object_type == "banana":
            arcade.draw_rectangle_filled(
                self.center_x,
                self.center_y,
                OBJECT_SIZE * 2,
                OBJECT_SIZE,
                arcade.color.YELLOW,
            )
            arcade.draw_rectangle_outline(
                self.area.center_x,
                self.area.center_y,
                OBJECT_SIZE * 2,
                OBJECT_SIZE,
                arcade.color.YELLOW,
            )
        elif self.object_type == "orange":
            arcade.draw_rectangle_filled(
                self.center_x,
                self.center_y,
                OBJECT_SIZE,
                OBJECT_SIZE,
                arcade.color.ORANGE,
            )
            arcade.draw_rectangle_outline(
                self.area.center_x,
                self.area.center_y,
                OBJECT_SIZE,
                OBJECT_SIZE,
                arcade.color.ORANGE,
            )
        elif self.object_type == "pear":
            arcade.draw_ellipse_filled(
                self.center_x,
                self.center_y,
                OBJECT_SIZE,
                OBJECT_SIZE * 1.5,
                arcade.color.GREEN,
            )
            arcade.draw_ellipse_outline(
                self.area.center_x,
                self.area.center_y,
                OBJECT_SIZE,
                OBJECT_SIZE * 1.5,
                arcade.color.GREEN,
            )


class TableArrangement(arcade.Window):
    def __init__(self, width, height, title):
        super().__init__(width, height, title)

        arcade.set_background_color(arcade.color.WHITE)

        self.robot = Robot(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)

        self.object_list = []

        apple = Object(TABLE_X + 50, TABLE_Y + TABLE_HEIGHT - 50, "apple")
        banana = Object(TABLE_X + TABLE_WIDTH - 50, TABLE_Y + 50, "banana")
        orange = Object(TABLE_X + 150, TABLE_Y + TABLE_HEIGHT - 100, "orange")
        pear = Object(TABLE_X + TABLE_WIDTH - 150, TABLE_Y + 100, "pear")

        self.object_list.append(apple)
        self.object_list.append(banana)
        self.object_list.append(orange)
        self.object_list.append(pear)

        # Define the areas for each shape
        apple.area = Object(TABLE_X + 100, TABLE_Y + TABLE_HEIGHT - 100, None)
        banana.area = Object(TABLE_X + TABLE_WIDTH - 100, TABLE_Y + 100, None)
        orange.area = Object(TABLE_X + 200, TABLE_Y + TABLE_HEIGHT - 150, None)
        pear.area = Object(TABLE_X + TABLE_WIDTH - 200, TABLE_Y + 150, None)

        self.selected_object = apple

    def on_draw(self):
        arcade.start_render()

        arcade.draw_rectangle_filled(
            TABLE_X + TABLE_WIDTH // 2,
            TABLE_Y + TABLE_HEIGHT // 2,
            TABLE_WIDTH,
            TABLE_HEIGHT,
            arcade.color.LIGHT_GRAY,
        )

        self.robot.draw()
        for obj in self.object_list:
            obj.draw()

    def update(self, delta_time):
        self.robot.perform_action(self.object_list)
        for obj in self.object_list:
            obj.update(self.object_list)

        # Check if all objects are in their correct areas
        all_in_areas = all(
            obj.area is not None and obj.intersects(obj.area)
            for obj in self.object_list
        )
        if all_in_areas:
            arcade.close_window()

    def move_object(self, object_type, direction):
        for obj in self.object_list:
            if obj.object_type == object_type:
                if direction == "up":
                    obj.change_y = 2
                elif direction == "down":
                    obj.change_y = -2
                elif direction == "left":
                    obj.change_x = -2
                elif direction == "right":
                    obj.change_x = 2

    def stop_object(self, object_type):
        for obj in self.object_list:
            if obj.object_type == object_type:
                obj.change_x = 0
                obj.change_y = 0

    def on_key_press(self, key, modifiers):
        if key == arcade.key.W:
            self.move_object(self.selected_object.object_type, "up")
        elif key == arcade.key.S:
            self.move_object(self.selected_object.object_type, "down")
        elif key == arcade.key.A:
            self.move_object(self.selected_object.object_type, "left")
        elif key == arcade.key.D:
            self.move_object(self.selected_object.object_type, "right")
        elif key == arcade.key.SPACE:
            index = self.object_list.index(self.selected_object)
            next_index = (index + 1) % len(self.object_list)
            self.selected_object = self.object_list[next_index]
            self.stop_object(self.selected_object.object_type)

    def on_key_release(self, key, modifiers):
        if key in (arcade.key.W, arcade.key.S, arcade.key.A, arcade.key.D):
            self.stop_object(self.selected_object.object_type)


def main():
    game = TableArrangement(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE)
    arcade.run()


if __name__ == "__main__":
    main()
