import torch
from pathlib import Path
import math
import pygame
import sys
import time
import os
from moviepy import *
import glob

"""
Prelims : setting up data paths
"""
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_path = Path("save")
assert save_path.exists(), "save folder does not exist"
data_dir = save_path / "7-21-LR"
assert data_dir.exists(), "data directory does not exist"
data_path = data_dir / "data.tar"
assert data_path.exists(), "data path does not exist"
data_dict = torch.load(data_path, weights_only=False, map_location=DEVICE)

left_right = True

# Set up output directory
os.makedirs("frames", exist_ok=True)

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

upper_triangle_coords = [(-1,2), (0,2), (1,2), (2,2), (0,1), (1,1), (2,1), (1,0), (2,0), (2,-1)]

all_states = data_dict['all_states']
episode_states = []
run = 0
run_states = all_states[run]
for episode in run_states[100:150]:
    check_upper_triangle = False
    i = 0
    all_agent_orig_state = []
    for step in episode:
        odor_indicator = 0
        check_odor_A = False
        check_no_odor = True
        agent_full_state = step
        agent_north_cart = [agent_full_state[3], agent_full_state[4], agent_full_state[5], agent_full_state[6]]
        agent_orig_state = conv_north_cartesian2orig(coords_orig=agent_north_cart)
        agent_coords = (agent_orig_state[0], agent_orig_state[1])
        if agent_coords in upper_triangle_coords:
            check_upper_triangle = True

        if agent_full_state[0].item() == 1:
            check_no_odor = True
            odor_indicator = 0
        elif agent_full_state[1].item() == 1:
            check_odor_A = True
            check_no_odor = False
            odor_indicator = 1
        else:
            check_odor_A = False
            check_no_odor = False
            odor_indicator = 2

        agent_orig_state.append(odor_indicator)
        all_agent_orig_state.append(agent_orig_state)
        i += 1

    states = [{"cue": s[3], "x": s[0].item(), "y": s[1].item(), "heading": s[2].item(), "UpperTriangle": check_upper_triangle} for s in all_agent_orig_state]

    for state in states:
        deg = state["heading"]
        state["heading"] = degrees_to_cardinal(deg)
    #print(states)

    episode_states.append(states)

print(f"NUM EPISODES: {len(episode_states)}")

# states = [
#     {"x": -2, "y": 1, "heading": "S"},
# ]








# Constants
GRID_SIZE = 5
CELL_SIZE = 100
WINDOW_SIZE = GRID_SIZE * CELL_SIZE
FPS = 3  # frames per second

# Colors
RED = (255, 0, 0)
BLACK = (0, 0, 0)

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

# Set up the font (name, size)
font = pygame.font.SysFont(None, 48)  # None means default font

# Loading Sprites
tile_image = pygame.image.load('sprites/tile100.png').convert()
tile_image = pygame.transform.scale(tile_image, (CELL_SIZE, CELL_SIZE))

odor_image = pygame.image.load('sprites/odor.png').convert_alpha()
odor_image = pygame.transform.scale(odor_image, (55,55))

reward_image = pygame.image.load('sprites/water_drop.png').convert_alpha()
reward_image = pygame.transform.scale(reward_image, (55,55))

mouse_sprites = {
    'N': pygame.image.load('sprites/mouse_up.png').convert_alpha(),
    'S': pygame.image.load('sprites/mouse_down.png').convert_alpha(),
    'W': pygame.image.load('sprites/mouse_left.png').convert_alpha(),
    'E': pygame.image.load('sprites/mouse_right.png').convert_alpha()
}

def grid_to_screen(x, y):
    screen_x = WINDOW_SIZE // 2 + x * CELL_SIZE
    screen_y = WINDOW_SIZE // 2 - y * CELL_SIZE  # invert y so +y is up
    return (screen_x, screen_y)

# Draw grid lines
def draw_grid(upper_triangle=True):
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            screen_x = col * CELL_SIZE
            screen_y = row * CELL_SIZE
            screen.blit(tile_image, (screen_x, screen_y))
    for x in range(0, WINDOW_SIZE, CELL_SIZE):
        pygame.draw.line(screen, (200, 200, 200), (x, 0), (x, WINDOW_SIZE))
    for y in range(0, WINDOW_SIZE, CELL_SIZE):
        pygame.draw.line(screen, (200, 200, 200), (0, y), (WINDOW_SIZE, y))


    if upper_triangle:
        # Draw staircase diagonal by highlighting the grid edges
        for i in range(GRID_SIZE):
            # Horizontal segment: move down by 1
            start_h = (i * CELL_SIZE, (i + 1) * CELL_SIZE)
            end_h = ((i + 1) * CELL_SIZE, (i + 1) * CELL_SIZE)
            pygame.draw.line(screen, BLACK, start_h, end_h, 3)

            # Vertical segment
            start_v = ((i + 1) * CELL_SIZE, (i + 1) * CELL_SIZE)
            end_v = ((i + 1) * CELL_SIZE, (i + 2) * CELL_SIZE)
            pygame.draw.line(screen, BLACK, start_v, end_v, 3)
    else:
        # Draw staircase shifted up by 1
        for i in range(1, GRID_SIZE + 1):
            # Horizontal segment: row i - 1
            start_h = ((i - 1) * CELL_SIZE, (i - 1) * CELL_SIZE)
            end_h = (i * CELL_SIZE, (i - 1) * CELL_SIZE)
            pygame.draw.line(screen, BLACK, start_h, end_h, 3)

            # Vertical segment: col i
            start_v = (i * CELL_SIZE, (i - 1) * CELL_SIZE)
            end_v = (i * CELL_SIZE, i * CELL_SIZE)
            pygame.draw.line(screen, BLACK, start_v, end_v, 3)

def draw_agent(state):
    x, y = state["x"], state["y"]
    heading = state['heading']
    sprite = mouse_sprites[heading]
    dx, dy = arrow_map[heading]
    center = grid_to_screen(x, y)

    rect = sprite.get_rect(center=center)
    screen.blit(sprite, rect)


# Main loop to animate states
agent = 0
i = 100
no_odor = True
odor_A = False
upper_triangle = False
num_frame = 0
# TODO: put a red X on incorrect reward port
# TODO: clearly mark the upper and lower triangle on the grid
for episode in episode_states:
    j = 0
    # Render the text (text, antialias, color)
    episode_text_surface = font.render(f"Episode {i}", True, (255, 0, 0))
    for state in episode:
        print(state)
        if (all_states[agent][i][j][0].item() == 1.0): # TODO: optimize using dictionary to store cue
            no_odor = True
            odor_label = 'None'
        elif (all_states[agent][i][j][1].item() == 1.0):
            no_odor = False
            odor_A = True
            odor_label = 'A'
        else:
            no_odor = False
            odor_A = False
            odor_label = 'B'

        odor_text_surface = font.render(f"Odor {odor_label}", True, (0, 0, 255))
        upper_triangle = state["UpperTriangle"]

        screen.fill((30, 30, 30))
        draw_grid(upper_triangle=upper_triangle)

        # TODO: put an 'X' on the incorrect reward port
        if no_odor == True:
            if upper_triangle == False:
                screen.blit(odor_image, (25, 425))
            else: 
                screen.blit(odor_image, (425, 25))
        elif odor_A == True and left_right == True:
            if upper_triangle == True:
                screen.blit(reward_image, (25, 25)) # WEST
            else:
                screen.blit(reward_image, (425, 425)) # EAST
        elif odor_A == False and left_right == True:
            if upper_triangle == True:
                screen.blit(reward_image, (425, 425)) # EAST
            else:
                screen.blit(reward_image, (25, 25)) # WEST

        draw_agent(state)

        # Draw the text at position 
        screen.blit(episode_text_surface, (305, 0))
        screen.blit(odor_text_surface, (0, 0))


        # Save to PNG with alpha
        pygame.image.save(screen, f"frames/frame_{num_frame:04d}.png")

        pygame.display.flip()
        clock.tick(FPS)  # slow down to visible speed

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Wait for spacebar to proceed
        # waiting_for_space = True
        # while waiting_for_space:
        #     for event in pygame.event.get():
        #         if event.type == pygame.QUIT:
        #             pygame.quit()
        #             sys.exit()
        #         elif event.type == pygame.KEYDOWN:
        #             if event.key == pygame.K_SPACE:
        #                 waiting_for_space = False
        j += 1
        num_frame += 1
    i += 1

# Wait before quitting
time.sleep(1)
pygame.quit()

# frames_path = "frames"
# # Get all .png files in the frames directory and sort them
# frame_files = sorted(glob.glob("frames/frame_*.png"))

# # Create clip (with alpha=True for transparency)
# clip = ImageSequenceClip(frame_files, fps=FPS, with_mask=True)

# # Save as .mov using PNG codec (supports alpha)
# clip.write_videofile("output.mov", codec="png", fps=FPS, write_logfile = True)