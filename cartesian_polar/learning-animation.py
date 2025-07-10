import torch
from pathlib import Path
import math
import pygame
import sys
import time

"""
Prelims : setting up data paths
"""
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_path = Path("save")
assert save_path.exists(), "save folder does not exist"
data_dir = save_path / "7-8-LR"
assert data_dir.exists(), "data directory does not exist"
data_path = data_dir / "data.tar"
assert data_path.exists(), "data path does not exist"
data_dict = torch.load(data_path, weights_only=False, map_location=DEVICE)

"""
Function for converting Cartesian North coords to Origin coords
"""
def conv_north_cartesian2orig(coords_orig):
        new_x = -coords_orig[0] + 2
        new_y = -coords_orig[1] + 2

        # cos and sin are switch beacuse `direction_orig` is taken from the north port
        sin_dir = -coords_orig[2]
        cos_dir = -coords_orig[3]
        new_direction = torch.atan2(input=sin_dir, other=cos_dir) * 180 / math.pi
        new_direction = new_direction % 360
        return [new_x, new_y, new_direction]

"""
Function for converting angle degree to cardinal direction
"""
def degrees_to_cardinal(degree):
    # Normalize the degree to [0, 360)
    degree = degree % 360

    # Define the mapping
    directions = {
        0: 'N',
        90: 'E',
        180: 'S',
        270: 'W'
    }

    # Find the closest cardinal angle
    closest = min(directions.keys(), key=lambda x: abs(x - degree))
    return directions[closest]



all_states = data_dict['all_states']
run = 0
run_states = all_states[run]
episode = run_states[0]
i = 0
all_agent_orig_state = []
for step in episode:
    agent_full_state = step
    agent_north_cart = [agent_full_state[3], agent_full_state[4], agent_full_state[5], agent_full_state[6]]
    agent_orig_state = conv_north_cartesian2orig(coords_orig=agent_north_cart)
    all_agent_orig_state.append(agent_orig_state)
    i += 1

states = [{"x": s[0].item(), "y": s[1].item(), "heading": s[2].item()} for s in all_agent_orig_state]

for state in states:
    deg = state["heading"]
    state["heading"] = degrees_to_cardinal(deg)
print(states)

# states = [
#     {"x": 0, "y": 0, "heading": "E"},
#     {"x": 1, "y": 0, "heading": "E"},
#     {"x": 2, "y": 0, "heading": "S"},
#     {"x": 2, "y": 1, "heading": "S"},
#     {"x": 2, "y": 2, "heading": "W"},
#     {"x": 1, "y": 2, "heading": "N"},
# ]

# Constants
GRID_SIZE = 5
CELL_SIZE = 100
WINDOW_SIZE = GRID_SIZE * CELL_SIZE
FPS = 2  # frames per second

# Map headings to direction arrows
arrow_map = {
    "N": (0, 1),
    "S": (0, -1),
    "E": (1, 0),
    "W": (-1, 0)
}

# Initialize pygame
pygame.init()
screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
pygame.display.set_caption("Agent Animation")
clock = pygame.time.Clock()

# Loading Sprites
tile_image = pygame.image.load('sprites/tile100.png').convert()
tile_image = pygame.transform.scale(tile_image, (CELL_SIZE, CELL_SIZE))


def grid_to_screen(x, y):
    screen_x = WINDOW_SIZE // 2 + x * CELL_SIZE
    screen_y = WINDOW_SIZE // 2 - y * CELL_SIZE  # invert y so +y is up
    return (screen_x, screen_y)

# Draw grid lines
def draw_grid():
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            screen_x = col * CELL_SIZE
            screen_y = row * CELL_SIZE
            screen.blit(tile_image, (screen_x, screen_y))
    for x in range(0, WINDOW_SIZE, CELL_SIZE):
        pygame.draw.line(screen, (200, 200, 200), (x, 0), (x, WINDOW_SIZE))
    for y in range(0, WINDOW_SIZE, CELL_SIZE):
        pygame.draw.line(screen, (200, 200, 200), (0, y), (WINDOW_SIZE, y))

def draw_agent(state):
    x, y = state["x"], state["y"]
    heading = state['heading']
    dx, dy = arrow_map[heading]
    center = grid_to_screen(x, y)
    pygame.draw.circle(screen, (0, 128, 255), center, 20)
    
    tip = (center[0] + dx * 30, center[1] - dy * 30)  # subtract dy to flip y
    pygame.draw.line(screen, (255, 255, 255), center, tip, 3)


# Main loop to animate states
for state in states:
    print(state)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    screen.fill((30, 30, 30))
    draw_grid()
    draw_agent(state)

    pygame.display.flip()
    clock.tick(FPS)  # slow down to visible speed

# Wait before quitting
time.sleep(1)
pygame.quit()

# PROCEEDING BY SPACE BAR

# state_index = 0
# running = True

# while running:
#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             running = False

#         # Advance to next state on spacebar press
#         elif event.type == pygame.KEYDOWN:
#             if event.key == pygame.K_SPACE:
#                 if state_index < len(states):
#                     # Draw next state
#                     screen.fill((30, 30, 30))
#                     draw_grid()
#                     draw_agent(states[state_index])
#                     print(states[state_index])
#                     pygame.display.flip()
#                     state_index += 1
#                 else:
#                     print("End of state list.")

# pygame.quit()