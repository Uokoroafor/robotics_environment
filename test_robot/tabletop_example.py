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
        return abs(self.center_x - obj.center_x) < (ROBOT_SIZE + OBJECT_SIZE) / 2 and abs(self.center_y - obj.center_y) < (ROBOT_SIZE + OBJECT_SIZE) / 2

class Object:
    def __init__(self, x, y, object_type):
        self.center_x = x
        self.center_y = y
        self.change_x = 0
        self.change_y = 0
        self.object_type = object_type

    def update(self):
        self.center_x += self.change_x
        self.center_y += self.change_y

    def draw(self):
        if self.object_type == "apple":
            arcade.draw_circle_filled(self.center_x, self.center_y, OBJECT_SIZE, arcade.color.RED)
        elif self.object_type == "banana":
            arcade.draw_rectangle_filled(self.center_x, self.center_y, OBJECT_SIZE * 2, OBJECT_SIZE, arcade.color.YELLOW)

class TableArrangement(arcade.Window):
    def __init__(self, width, height, title):
        super().__init__(width, height, title)

        arcade.set_background_color(arcade.color.WHITE)

        self.robot = Robot(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)

        self.object_list = []

        apple = Object(TABLE_X + 50, TABLE_Y + TABLE_HEIGHT - 50, "apple")
        banana = Object(TABLE_X + TABLE_WIDTH - 50, TABLE_Y + 50, "banana")
        self.object_list.append(apple)
        self.object_list.append(banana)

        self.selected_object = None

    def on_draw(self):
        arcade.start_render()

        arcade.draw_rectangle_filled(TABLE_X + TABLE_WIDTH // 2, TABLE_Y + TABLE_HEIGHT // 2, TABLE_WIDTH, TABLE_HEIGHT, arcade.color.LIGHT_GRAY)

        self.robot.draw()
        for obj in self.object_list:
            obj.draw()

    def update(self, delta_time):
        self.robot.perform_action(self.object_list)
        for obj in self.object_list:
            obj.update()

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

def main():
    game = TableArrangement(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE)
    arcade.run()

if __name__ == "__main__":
    main()
