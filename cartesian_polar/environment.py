"""Environment definition."""

import math
from collections import OrderedDict
from enum import Enum

import torch
import torch.nn.functional as F
from tensordict.tensordict import TensorDict

# !TODO Investigate relative import issue
from utils import make_deterministic, random_choice


class OdorCondition(Enum):
    """Keeps track before and after the odor has been presented."""

    pre = 1
    post = 2


class Cues(Enum):
    """List the possible cues conditions."""

    """
    NoOdor corresponds to when the agent has not reached an odor port yet
    """

    NoOdor = 0
    OdorA = 1
    OdorB = 2


class Ports(Enum):
    """List the coordinates of the 4 ports."""

    North = (2, 2)
    South = (-2, -2)
    West = (-2, 2)
    East = (2, -2)


class OdorPorts(Enum):
    """List the odor ports."""

    North = Ports.North.value
    South = Ports.South.value


class Actions(Enum):
    """List the possible actions."""

    forward = 0
    left = 1
    right = 2
    # backward = 3


class TriangleState(Enum):
    """Keeps track of which part of the arena the agent is located."""

    upper = 1
    lower = 2


class TaskID(Enum):
    """List the different tasks."""

    EastWest = 1
    LeftRight = 2


CONTEXTS_LABELS = OrderedDict(
    [
        (Cues.NoOdor, "Pre odor"),
        (Cues.OdorA, "Odor A"),
        (Cues.OdorB, "Odor B"),
    ]
)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Environment:
    """Environment logic."""

    def __init__(self, taskid, seed=None):
        self.taskid = eval(f"TaskID.{taskid}")
        if seed: # If you have a seed, random number generator is deterministic
            self.generator = make_deterministic(seed=seed)
        else:
            self.generator = None
        # self.rows = 5
        # self.cols = 5
        # self.origX = 0
        # self.origY = 0
        """
        - The environment is a 5x5 2D grid, (X: -2 to 2, Y: -2 to 2)
        - Grid spacing is 1 unit
        - Agent initialized in either lower or upper triangle
        """
        self.tile_step = 1
        self.rangeX = {"min": -2, "max": 2}
        self.rangeY = {"min": -2, "max": 2}
        self.head_angle_space = torch.tensor( # Head orientation of the agent
            [0, 90, 180, 270], device=DEVICE
        )  # In degrees, 0 DEGREES IS NORTH
        """
        Creating tensors for every possible tile coordinate
        """
        self.tiles_locations = {
            "x": torch.arange(
                start=self.rangeX["min"],
                end=self.rangeX["max"] + 1,
                step=self.tile_step,
                device=DEVICE,
            ),
            "y": torch.arange(
                start=self.rangeY["min"],
                end=self.rangeY["max"] + 1,
                step=self.tile_step,
                device=DEVICE,
            ),
        }
        # self.cues = [*LightCues, *OdorID]

        """
        Defining the action space : set of all possible actions
        """
        self.action_space = torch.tensor(
            [item.value for item in Actions], device=DEVICE
        )
        # self.action_space = ActionSpace()
        # self.numActions = len(self.action_space())
        self.numActions = len(self.action_space)

        # self.state_space = {
        #     "location": self.tiles_locations,
        #     # "cue": set(OdorID).union(LightCues),
        #     "cue": set(Cues),
        # }
        # self.numStates = tuple(len(item) for item in self.state_space.values())
        self.reset()

    """
    Initializing the environment
    """
    def reset(self):
        """
        Reset the environment.

        Randomizes which triangle part of the arena the agent is located in
        - corresponds to one of the two odor ports (N / S)
        """
        self.TriangleState = TriangleState(
            random_choice(
                torch.tensor([item.value for item in TriangleState], device=DEVICE),
                generator=self.generator,
            ).item()
        )
        # self.head_direction = random_choice(
        #     self.head_angle_space,
        #     generator=self.generator,
        # )
        # start_state = torch.cat(
        #     (
        #         torch.tensor(Cues.NoOdor.value, device=DEVICE).unsqueeze(-1),
        #         self.agent_coords,
        #     )
        # )
        """
        Randomizes agent position : agent starts on lower or upper triangle
        """
        agent_coords = self.sample_coord_position()
        self.current_state = TensorDict(
            {
                # Cue always starts as NoOdor
                "cue": torch.tensor(Cues.NoOdor.value, device=DEVICE).unsqueeze(-1), 
                "x": agent_coords[0].unsqueeze(-1),
                "y": agent_coords[1].unsqueeze(-1),
                "direction": random_choice( # Head direction is random
                    self.head_angle_space,
                    generator=self.generator,
                ),
            },
            batch_size=[1],
            device=DEVICE,
        )
        # At first, the agent is not exposed to the odor
        self.odor_condition = OdorCondition.pre 
        """
        The odor that is presented is random
        """
        self.odor_ID = Cues(
            random_choice(
                torch.tensor(
                    [item.value for item in Cues if item.name != "NoOdor"],
                    device=DEVICE,
                ),
                generator=self.generator,
            ).item()
        )
        # self.TaskID = TaskID(
        #     random_choice(
        #         torch.tensor([item.value for item in TaskID], device=DEVICE),
        #         generator=self.generator,
        #     ).item()
        # )
        return self.current_state

    def sample_coord_position(self):
        """Sample coordinates X and Y until they fit the upper or lower triangle."""
        while True:
            x_sampled = (
                torch.distributions.Uniform(self.rangeX["min"], self.rangeX["max"])
                .sample()
                .round()
            )
            y_sampled = (
                torch.distributions.Uniform(self.rangeY["min"], self.rangeY["max"])
                .sample()
                .round()
            )
            if (
                self.TriangleState == TriangleState.upper
                and x_sampled + y_sampled >= 0
                or self.TriangleState == TriangleState.lower
                and x_sampled + y_sampled <= 0
            ):
                break
        coords = torch.tensor([x_sampled, y_sampled], device=DEVICE)
        return coords

    def is_terminated(self, state):
        """Return if the episode is terminated or not."""
        is_terminated = False
        if self.odor_condition == OdorCondition.post and ( # Agent has smelled the odor
            # Checks if agent has gone to one of the reward ports
            (state["x"], state["y"]) == Ports.West.value
            and state["direction"] in [0, 270]
            or (state["x"], state["y"]) == Ports.East.value
            and state["direction"] in [90, 180]
        ):
            is_terminated = True # Then terminates
        return is_terminated

    def reward(self, state):
        """Observe the reward."""
        reward = 0
        if self.odor_condition == OdorCondition.post:
            if self.taskid == TaskID.EastWest:
                """
                - Checks if the agent is in the proper reward port for the cue
                - Cue A always has reward in the West, Cue B in the East
                """
                if (
                    state["cue"] == Cues.OdorA.value
                    and (state["x"], state["y"]) == Ports.West.value
                    and state["direction"] in [0, 270]
                    or state["cue"] == Cues.OdorB.value
                    and (state["x"], state["y"]) == Ports.East.value
                    and state["direction"] in [90, 180] 
                ):
                    reward = 1
            elif self.taskid == TaskID.LeftRight:
                """
                If A, always turn to the left
                """
                if self.TriangleState == TriangleState.upper:
                    if (
                        state["cue"] == Cues.OdorA.value
                        and (state["x"], state["y"]) == Ports.West.value
                        and state["direction"] in [0, 270]
                        or state["cue"] == Cues.OdorB.value
                        and (state["x"], state["y"]) == Ports.East.value
                        and state["direction"] in [90, 180]
                    ):
                        reward = 1
                elif self.TriangleState == TriangleState.lower:
                    if (
                        state["cue"] == Cues.OdorA.value
                        and (state["x"], state["y"]) == Ports.East.value
                        and state["direction"] in [90, 180]
                        or state["cue"] == Cues.OdorB.value
                        and (state["x"], state["y"]) == Ports.West.value
                        and state["direction"] in [0, 270]
                    ):
                        reward = 1
                else:
                    raise ValueError("Impossible value, must be `upper` or `lower`")
            else:
                raise ValueError(
                    "Impossible `TaskID` value, must be `EastWest` or `LeftRight`"
                )
        return reward

    def step(self, action, current_state, use_internal_state=None):
        """Take an action, observe reward and the next state."""
        new_state = TensorDict({}, batch_size=[1], device=DEVICE)
        if use_internal_state:
            current_state = self.current_state
        new_state["cue"] = current_state["cue"]
        new_agent_loc = self.move( # Defines the new location of the agent
            x=current_state["x"],
            y=current_state["y"],
            direction=current_state["direction"],
            action=action,
            step=self.tile_step,
        )
        # Update the location state of the agent
        new_state["x"] = new_agent_loc[0].unsqueeze(-1)
        new_state["y"] = new_agent_loc[1].unsqueeze(-1)
        new_state["direction"] = new_agent_loc[2].unsqueeze(-1)

        # Update internal states
        # Checking if the agent is now at one of the odor ports
        if (new_state["x"].item(), new_state["y"].item()) in { 
            item.value for item in OdorPorts
        }:
            # If so, the agent has smelled the odor
            self.odor_condition = OdorCondition.post
            # Set the cue to the odor ID 
            new_state["cue"] = torch.tensor([self.odor_ID.value], device=DEVICE) 

        # Checking if reward is received in the new state
        reward = self.reward(new_state) 
        # Checking if the task is complete in the new state
        done = self.is_terminated(new_state) 
        return new_state, reward, done

    # def move(self, row, col, a):
    #     """Where the agent ends up on the map."""
    #     row_max, col_max = self.wall(row, col, a)
    #     if a == Actions.left.value:
    #         col = torch.tensor([col - 1, col_max], device=DEVICE).max()
    #     elif a == Actions.down.value:
    #         row = torch.tensor([row + 1, row_max], device=DEVICE).min()
    #     elif a == Actions.right.value:
    #         col = torch.tensor([col + 1, col_max], device=DEVICE).min()
    #     elif a == Actions.up.value:
    #         row = torch.tensor([row - 1, row_max], device=DEVICE).max()
    #     return (row, col)

    def move(self, x, y, direction, action, step=1):
        """
        Where the agent ends up on the map.

        - When the agent moves, head direction moves by 90 degrees as well
        """
        def LEFT(x, y):
            """Move left in the allocentric sense."""
            x_min = ( # Defines the left boundary
                self.rangeX["min"] if self.TriangleState == TriangleState.lower else -y 
            )
            x = max(x - step, x_min)
            angle = 270 # 270 degrees defines west
            return angle, x

        def DOWN(x, y):
            """Move down in the allocentric sense."""
            y_min = (
                self.rangeY["min"] if self.TriangleState == TriangleState.lower else -x
            )
            y = max(y - step, y_min)
            angle = 180
            return angle, y

        def RIGHT(x, y):
            """Move right in the allocentric sense."""
            x_max = (
                self.rangeX["max"] if self.TriangleState == TriangleState.upper else -y
            )
            x = min(x + step, x_max)
            angle = 90
            return angle, x

        def UP(x, y):
            """Move up in the allocentric sense."""
            y_max = (
                self.rangeY["max"] if self.TriangleState == TriangleState.upper else -x
            )
            y = min(y + step, y_max)
            angle = 0
            return angle, y

        """
        Depending on what direction the agent is facing, 
        each action corresponds to a different relative egocentric movement
        """
        if direction.round() == 0:  # Facing north
            if action == Actions.left.value:
                direction, x = LEFT(x=x, y=y)
            elif action == Actions.right.value:
                direction, x = RIGHT(x=x, y=y)
            elif action == Actions.forward.value:
                direction, y = UP(x=x, y=y)
            # elif action == Actions.backward.value:
            #     direction, y = DOWN(x=x, y=y)
            #     direction = 0

        elif direction.round() == 90:  # Facing east
            if action == Actions.left.value:
                direction, y = UP(x=x, y=y)
            elif action == Actions.right.value:
                direction, y = DOWN(x=x, y=y)
            elif action == Actions.forward.value:
                direction, x = RIGHT(x=x, y=y)
            # elif action == Actions.backward.value:
            #     direction, x = LEFT(x=x, y=y)
            #     direction = 90

        elif direction.round() == 180:  # Facing south
            if action == Actions.left.value:
                direction, x = RIGHT(x=x, y=y)
            elif action == Actions.right.value:
                direction, x = LEFT(x=x, y=y)
            elif action == Actions.forward.value:
                direction, y = DOWN(x=x, y=y)
            # elif action == Actions.backward.value:
            #     direction, y = UP(x=x, y=y)
            #     direction = 180

        elif direction.round() == 270:  # Facing west
            if action == Actions.left.value:
                direction, y = DOWN(x=x, y=y)
            elif action == Actions.right.value:
                direction, y = UP(x=x, y=y)
            elif action == Actions.forward.value:
                direction, x = LEFT(x=x, y=y)
            # elif action == Actions.backward.value:
            #     direction, x = RIGHT(x=x, y=y)
            #     direction = 270

        return torch.tensor([x, y, direction], device=DEVICE)

    # def wall(self, x, y, a):
    #     """Get wall according to current location and next action."""
    #     if self.TriangleState == self.TriangleState.upper:
    #         if a == Actions.left.value:
    #             x = y
    #         elif a == Actions.down.value:
    #             y = x
    #         elif a == Actions.right.value:
    #             x = self.cols - 1
    #         elif a == Actions.up.value:
    #             y = 0

    #     elif self.TriangleState == self.TriangleState.lower:
    #         if a == Actions.left.value:
    #             x = 0
    #         elif a == Actions.down.value:
    #             y = self.rows - 1
    #         elif a == Actions.right.value:
    #             x = y
    #         elif a == Actions.up.value:
    #             y = x

    #     else:
    #         raise ValueError("Impossible value, must be `upper` or `lower`")
    #     return (y, x)

class DuplicatedCoordsEnv(Environment):
    """
    Wrap the base Environment class.

    Results in numerical only state space
    - this is how we take the Environment Cue, X&Y, and head direction
      and translate to an input array of 19 values
      (w/ Cartesian and Polar coords)
    """

    """
    TODO:
    - Make sure to have something in code that records where agent is all the time,
    not dependent on Cartesian
    - The code needs to be refactored, as you cannot silence the Cartesian coords
    """

    def __init__(self, taskid, seed=None):
        # Initialize the base class to get the base properties
        super().__init__(taskid=taskid, seed=seed)
        self.reset()

    # Dict: the original Environment info
    # Flat: the array of 19 inputs
    def conv_dict_to_flat_duplicated_coords(self, state):
        """Convert composite state dictionary to a tensor."""
        coords_orig = torch.tensor(
            [state["x"], state["y"], state["direction"]], device=DEVICE
        )
        cue = F.one_hot(state["cue"], num_classes=len(Cues)).squeeze()
        north_cart_coords = self.conv2north_cartesian(coords_orig)
        south_cart_coords = self.conv2south_cartesian(coords_orig)
        north_polar_coords = self.conv2north_polar(coords_orig)
        south_polar_coords = self.conv2south_polar(coords_orig)
        # This tensor is the format for network inputs
        conv_state = torch.cat(
            (
                cue,
                north_cart_coords,
                south_cart_coords,
                north_polar_coords,
                south_polar_coords,
            )
        )
        return conv_state

    def conv_flat_duplicated_coords_to_dict(self, state):
        """Convert back tensor state to original composite state."""
        coords_orig = self.conv_north_cartesian2orig(state[3:7])
        conv_state = TensorDict(
            {
                "cue": torch.tensor(
                    [state[0 : len(Cues)].argwhere().item()], device=DEVICE
                ),
                "x": torch.tensor([coords_orig[0]], device=DEVICE),
                "y": torch.tensor([coords_orig[1]], device=DEVICE),
                "direction": torch.tensor([coords_orig[2]], device=DEVICE),
            },
            batch_size=[1],
            device=DEVICE,
        )
        return conv_state

    def step(self, action, current_state, use_internal_state=None):
        """Wrap the base method."""
        current_conv_state = self.conv_flat_duplicated_coords_to_dict(current_state)
        new_state, reward, done = super().step(
            action, current_conv_state, use_internal_state=use_internal_state
        )
        new_state_conv = self.conv_dict_to_flat_duplicated_coords(new_state)
        reward = torch.tensor([reward], device=DEVICE).float()  # Convert to tensor
        new_state_conv = (
            new_state_conv.float()
        )  # Cast to float for PyTorch multiplication

        return new_state_conv, reward, done

    def reset(self):
        """Wrap the base method."""
        state = super().reset()
        conv_state = self.conv_dict_to_flat_duplicated_coords(state)
        conv_state = conv_state.float()  # Cast to float for PyTorch multiplication
        return conv_state

    def conv2north_cartesian(self, coords_orig):
        """
        Convert origin (0, 0) coords to Cartesian coords from the North port.
        This sets the North port as (0, 0)
        """
        new_x = -coords_orig[0] + 2
        new_y = -coords_orig[1] + 2
        # direction_orig = coords_orig[2].item()
        # if 0 <= direction_orig < 270:
        #     new_absolute_direction = 270 - direction_orig
        # elif 270 <= direction_orig < 360:
        #     new_absolute_direction = 270 + 360 - direction_orig
        # else:
        #     raise ValueError("Impossible angular value")
        # new_absolute_direction = (
        #     0 if new_absolute_direction == 360 else new_absolute_direction
        # )
        _, _, south_cos, south_sin = self.conv2south_cartesian(coords_orig)
        cos_dir = -south_cos
        sin_dir = -south_sin
        return torch.tensor([new_x, new_y, cos_dir, sin_dir], device=DEVICE)

    def conv2south_cartesian(self, coords_orig):
        """Convert origin (0, 0) coords to Cartesian coords from the South port."""
        new_x = coords_orig[0] + 2
        new_y = coords_orig[1] + 2
        # direction_orig = coords_orig[2].item()
        direction_orig = coords_orig[2]
        # if 0 <= direction_orig <= 90:
        #     new_absolute_direction = 90 - direction_orig
        # else:
        #     new_absolute_direction = 450 - direction_orig

        # cos and sin are switch beacuse `direction_orig` is taken from the north port
        cos_dir = torch.sin(direction_orig * math.pi / 180)
        sin_dir = torch.cos(direction_orig * math.pi / 180)
        return torch.tensor([new_x, new_y, cos_dir, sin_dir], device=DEVICE)

    def cartesian2polar(self, coords_orig):
        """Convert coordinates from Cartesian to polar."""
        length = torch.sqrt(coords_orig[0] ** 2 + coords_orig[1] ** 2)
        alpha = torch.atan2(
            input=coords_orig[1], other=coords_orig[0]
        )  # * 180 / math.pi
        cos_alpha = torch.cos(alpha)
        sin_alpha = torch.sin(alpha)

        # Head direction
        cos_orig = coords_orig[2]
        sin_orig = coords_orig[3]
        direction = torch.atan2(input=sin_orig, other=cos_orig)
        if alpha.sign() == direction.sign():
            angle_diff = direction - alpha
        else:
            # angles = [alpha, direction]
            # angle_sign = np.sign(angles[np.argmax(np.abs(angles))])
            # angle_diff = angle_sign*(abs(direction)+abs(alpha))
            angle_diff = direction.sign() * (abs(direction) + abs(alpha))
        cos_dir = torch.cos(angle_diff)
        sin_dir = torch.sin(angle_diff)
        return length, cos_alpha, sin_alpha, cos_dir, sin_dir

    def conv2north_polar(self, coords_orig):
        """Convert from origin (0, 0) coords to polar coords from the North port."""
        north_coords = self.conv2north_cartesian(coords_orig)
        # length, alpha = self.cartesian2polar(north_coords)
        length, cos_alpha, sin_alpha, cos_dir, sin_dir = self.cartesian2polar(
            north_coords
        )
        # direction_orig = coords_orig[2].item()
        # if 0 <= direction_orig <= 180:
        #     new_relative_direction = 270 - alpha - direction_orig
        # elif 180 < direction_orig <= 270:
        #     if direction_orig + alpha - 180 <= 90:
        #         new_relative_direction = 270 - alpha - direction_orig
        #     else:
        #         new_relative_direction = -(alpha - (270 - direction_orig))
        # elif 270 < direction_orig < 360:
        #     new_relative_direction = -(direction_orig - 270 + alpha)
        # else:
        #     raise ValueError("Impossible angular value")
        # if abs(new_relative_direction) > 180:
        #     new_relative_direction = new_relative_direction % 360
        # north_polar = torch.tensor(
        #     [length, alpha, new_relative_direction], device=DEVICE
        # )
        # cos_dir = torch.cos(new_relative_direction*math.pi/180)
        # sin_dir = torch.sin(new_relative_direction*math.pi/180)
        north_polar = torch.tensor(
            [length, cos_alpha, sin_alpha, cos_dir, sin_dir], device=DEVICE
        )
        return north_polar

    def conv2south_polar(self, coords_orig):
        """Convert from origin (0, 0) coords to polar coords from the South port."""
        south_coords = self.conv2south_cartesian(coords_orig)
        # length, alpha = self.cartesian2polar(south_coords)
        length, cos_alpha, sin_alpha, cos_dir, sin_dir = self.cartesian2polar(
            south_coords
        )
        # direction_orig = coords_orig[2].item()
        # if 0 <= direction_orig <= 90:
        #     if direction_orig + alpha <= 90:
        #         new_relative_direction = 90 - alpha - direction_orig
        #     else:
        #         new_relative_direction = -abs(90 - alpha - direction_orig)
        # else:
        #     new_relative_direction = -(direction_orig - 90 + alpha)
        # new_relative_direction = (
        #     abs(new_relative_direction)
        #     if new_relative_direction == -180
        #     else new_relative_direction
        # )
        # if abs(new_relative_direction) > 180:
        #     new_relative_direction = new_relative_direction % 360
        # cos_dir = torch.cos(new_relative_direction*math.pi/180)
        # sin_dir = torch.sin(new_relative_direction*math.pi/180)
        south_polar = torch.tensor(
            [length, cos_alpha, sin_alpha, cos_dir, sin_dir], device=DEVICE
        )
        return south_polar

    def conv_north_cartesian2orig(self, coords_orig):
        """Convert Cartesian coords from North port to origin (0, 0) coords."""
        new_x = -coords_orig[0] + 2
        new_y = -coords_orig[1] + 2

        # cos and sin are switch beacuse `direction_orig` is taken from the north port
        sin_dir = -coords_orig[2]
        cos_dir = -coords_orig[3]
        new_direction = torch.atan2(input=sin_dir, other=cos_dir) * 180 / math.pi
        new_direction = new_direction % 360
        # if 0 <= direction < 270:
        #     new_direction = 270 - direction
        # elif 270 <= direction < 360:
        #     new_direction = 270 + 360 - direction
        # else:
        #     raise ValueError("Impossible angular value")
        # new_direction = (
        #     0 if new_direction == 360 else new_direction
        # )
        return torch.tensor([new_x, new_y, new_direction], device=DEVICE)
