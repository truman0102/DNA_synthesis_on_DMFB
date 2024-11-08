import numpy as np
from typing import Tuple, List, Union
from dna_on_dmfb.enviroment.constants import VALID_NAMES, MOVEMENTS_2D, MOVEMENTS_4D
from copy import deepcopy
class Unit:
    def __init__(
            self,
            name: str,
            width: int,
            height: int,
            location: Union[Tuple[int, int, int, int], Tuple[int, int], List[int], np.ndarray], # type: ignore
            mobile: bool, # True for droplets, False for fixed regions
            **kwargs
        ) -> None:
        self.name = name
        self.width = width
        self.height = height
        assert len(location) in [2, 4], f"Location should be a tuple of 2 or 4 integers, got {location}"
        self._set_location(location)
        self.mobile = mobile
        self.kwargs = kwargs

    def _set_location(self, location: Union[Tuple[int, int], List[int], np.ndarray]) -> None:
        self.location = np.array(location) if not isinstance(location, np.ndarray) else location
        if len(self.location) == 2:
            self._mark_4D = False
            # self.center = self.location + np.array([self.width, self.height]) / 2.0
        else:
            self._mark_4D = True
            self.height = self.location[2] - self.location[0] + 1
            self.width = self.location[3] - self.location[1] + 1
            # self.center = np.array([self.location[0] + self.location[2], self.location[1] + self.location[3]]) / 2.0

    def __repr__(self) -> str:
        return f"{self.name} at {self.location} with size {self.width}x{self.height}"
    
    def __eq__(self, other) -> bool:
        return self.name == other.name and np.array_equal(self.location, other.location)
    
    def __hash__(self) -> int:
        return hash((f"{self.name}_{self.location[0]}_{self.location[1]}_{self.width}_{self.height}"))
    
    @property
    def center(self):
        if self._mark_4D:
            return np.array([self.location[0] + self.location[2], self.location[1] + self.location[3]]) / 2.0
        return self.location + np.array([self.width, self.height]) / 2.0

    @property
    def position(self):
        return self.location

    @property
    def x_upper_left(self):
        return int(self.location[0])
    @property
    def y_upper_left(self):
        return int(self.location[1])
    @property
    def x_lower_right(self):
        return int(self.location[0] + self.width - 1)
    @property
    def y_lower_right(self):
        return int(self.location[1] + self.height - 1)
    @property
    def bounding_box(self):
        return (self.x_upper_left, self.y_upper_left, self.x_lower_right, self.y_lower_right)
    @property
    def extended_boundaries(self):
        return (self.x_upper_left - 1, self.y_upper_left - 1, self.x_lower_right + 1, self.y_lower_right + 1)

class Entrance(Unit):
    
    def __init__(
            self,
            name,
            width,
            height ,
            location,
            **kwargs
        ) -> None:
        super().__init__(name, width, height, location, False, **kwargs)
        assert self.name in VALID_NAMES, f"Invalid name {self.name}"
        self.children = []

    def add_parent(self, parent):
        raise ValueError("Entrance cannot have parents")
    
    def add_child(self, child):
        self.children.append(child)

    def sent(self, goal):
        if self.name.startswith("d"): # dNTP or ddNTP
            if self.name.startswith("dd"): # ddNTP
                return ddNTP(self.name, self.width, self.height, deepcopy(self.location), goal)
            return dNTP(self.name, self.width, self.height, deepcopy(self.location), goal)
        return Droplet(self.name, self.width, self.height, deepcopy(self.location), goal)

def get_nucleotide_from_name(name: str):
    # from dNTP or ddNTP to N
    if name.startswith("d"):
        for s in name:
            if s in "ATCG":
                return s
    elif name[0] in "ATCG":
        return name.split("_")[0]
    raise ValueError(f"Invalid nucleotide name {name}")
        
def common_prefix(s1, s2):
    res = ""
    for i in range(min(len(s1), len(s2))):
        if s1[i] == s2[i]:
            res += s1[i]
        else:
            break
    return res

class Zone(Unit):
    def __init__(
            self,
            name: str,
            width: int,
            height: int,
            location: Union[Tuple[int, int, int, int], Tuple[int, int], List[int], np.ndarray], # type: ignore
            goal: str,
            level: int = 0,
            **kwargs
        ) -> None:
        super().__init__(name, width, height, location, False, **kwargs)
        self.parents = []
        self.children = []
        self.contents = set()
        self.level = level
        self.goal = goal # goal is a string that represents the goal production of the zone, a sequence of NTPs
        self.intermediate = "" # content is the current production of the zone, a sequence of NTPs
        self.mutate_prefixs = {}
        self.waiting_for_mutate = False

    def merge(self):
        if self.finished:
            raise ValueError("Zone is already finished")
        nuc = self._get_nucleotide
        if self.level == 1:
            assert "primer" in self.contents, "Primer is not present in the zone"
            self.contents = set(["primer"])
        else:
            self.contents.clear()
        self.intermediate += get_nucleotide_from_name(nuc)
        if self.intermediate in self.mutate_prefixs:
            self.waiting_for_mutate = True
            print(f"Zone {self.name} is waiting for mutation to {self.mutate_prefixs[self.intermediate]}")
        if self.finished:
            print(f"Zone {self.name} finished with {self.intermediate}")
        else:
            print(f"Zone {self.name} merged with {nuc} to produce {self.intermediate}")
    
    def mutate(self):
        assert self.to_mutate, "Zone is not ready to mutate"
        if self.level == 1:
            assert "eluent" in self.contents, "Eluent is not present in the zone"
            self.contents.remove("eluent")
        self.waiting_for_mutate = False
    
    def add_droplet(self, droplet):
        assert droplet.reach_goal, f"Droplet {droplet.name} is not at its goal, so it cannot be added to the zone"
        self.contents.add(droplet.class_name)
        if not self.waiting_for_mutate and self.to_merge:
            self.merge()
            return True
        return False

    def add_parent(self, parent):
        if self.level == 0:
            raise ValueError("Zone level is not set")
        elif self.level == 1:
            raise ValueError("Root zone cannot have parents")
        elif parent.level != self.level - 1:
            raise ValueError("Parent zone level is not compatible")
        self.parents.append(parent)

    def add_child(self, child):
        self.children.append(child)
        prefix = common_prefix(self.goal, child.goal)
        if prefix not in self.mutate_prefixs:
            self.mutate_prefixs[prefix] = []
        self.mutate_prefixs[prefix].append(child.goal)
    
    @property
    def _get_nucleotide(self):
        for s in self.contents:
            if s[0] in "dATCG":
                return s.split("_")[0]

    def is_goal_of(self, g: Union[Tuple[int, int, int, int], List[int], np.ndarray]): # type: ignore
        """
        Check if the given goal is a subset of the zone
        """
        return all(
            [self.x_upper_left <= g[0],
            self.y_upper_left <= g[1],
            self.x_lower_right >= g[2],
            self.y_lower_right >= g[3]]
        )
    
    @property
    def to_mutate(self):
        if self.level == 1:
            return self.waiting_for_mutate and "eluent" in self.contents
        return self.waiting_for_mutate

    @property
    def to_merge(self):
        if self.finished:
            return False
        if self.level > 1 and len(self.intermediate) == 0 and len(self.contents) > 0:
            return True
        base = self.next_nucleotide in self.contents
        if len(self.intermediate) == 0 and self.level == 1:
            return base and "primer" in self.contents
        return base

    @property
    def next_nucleotide(self):
        if len(self.intermediate) == len(self.goal) - 1:
            return "dd"+self.goal[-1]+"TP"
        elif len(self.intermediate) < len(self.goal):
            return "d"+self.goal[len(self.intermediate)]+"TP"
        return None
    
    @property
    def need_primer(self):
        return self.level == 1 and len(self.intermediate) == 0 and "primer" not in self.contents
    
    @property
    def need_eluent(self):
        return self.level == 1 and "eluent" not in self.contents and self.intermediate in self.mutate_prefixs

    @property
    def next_level(self):
        return self.level + 1
    
    @property
    def finished(self):
        return self.intermediate.__eq__(self.goal)

class Droplet(Unit):
    def __init__(
            self,
            name: str,
            width: int,
            height: int,
            location: Union[Tuple[int, int], List[int], np.ndarray], # (x_upper_left, y_upper_left)
            goal: Union[Tuple[int, int, int, int], List[int], np.ndarray, Zone, str], # (x_upper_left, y_upper_left, x_lower_right, y_lower_right)
            **kwargs
        ) -> None:
        super().__init__(name, width, height, location, True, **kwargs)
        assert len(goal) == 4 or isinstance(goal, (Zone, str)), f"Goal should be a tuple of 4 integers, got {goal}"
        self._set_goal(goal)
        self.last_area = None
        self.last_movement = None

    def move(self, movement: Union[Tuple[int, int, int, int], Tuple[int, int], List[int], np.ndarray]) -> None:
        if not self.mobile:
            return
        assert len(movement) == len(self.location), f"Movement should be a tuple of {len(self.location)} integers, got {movement}"
        # assert movement in MOVEMENTS_2D, f"Invalid movement {movement}"
        # get last area in the form of (x_upper_left, y_upper_left, x_lower_right, y_lower_right)
        if movement[0] == -1: # UP
            self.last_area = (self.x_lower_right, self.y_upper_left, self.x_lower_right, self.y_lower_right)
        elif movement[0] == 1: # DOWN
            self.last_area = (self.x_upper_left, self.y_upper_left, self.x_upper_left, self.y_lower_right)
        elif movement[1] == -1: # LEFT
            self.last_area = (self.x_upper_left, self.y_lower_right, self.x_lower_right, self.y_lower_right)
        elif movement[1] == 1: # RIGHT
            self.last_area = (self.x_upper_left, self.y_upper_left, self.x_lower_right, self.y_upper_left)
        movement = np.array(movement) if not isinstance(movement, np.ndarray) else movement
        self.location += movement
        is_reverse_action = np.abs((self.last_movement + movement).sum()) == 0 if self.last_movement is not None else False
        self.last_movement = movement
        return is_reverse_action

    def is_absolutely_safe(self, obstacles: np.ndarray) -> bool:
        assert obstacles.ndim == 2, f"Obstacles should be a 2D numpy array, got {obstacles.ndim}D"
        # TODO
        return not np.any(obstacles[self.x_upper_left:self.x_lower_right+1, self.y_upper_left:self.y_lower_right+1])

    def set_color(self, color: Union[Tuple[int, int, int], List[int], np.ndarray]) -> None:
        assert len(color) == 3, f"Color should be a tuple of 3 integers, got {color}"
        color = list(color)
        if max(color) > 1:
            self.color_integer = color
            self.color_float = [c / 255.0 for c in color]
            self.color_rgba = self.color_float + [1.0]
        else:
            self.color_float = color
            self.color_integer = [int(c * 255) for c in color]
            self.color_rgba = color + [1.0]

    def _set_goal(self, goal: Union[Tuple[int, int, int, int], List[int], np.ndarray]) -> None:
        if len(goal) == 4:
            self.goal = np.array(goal) if not isinstance(goal, np.ndarray) else goal
            self.goal_center = np.array([self.goal[0] + self.goal[2], self.goal[1] + self.goal[3]]) / 2.0
        elif isinstance(goal, (Zone, Entrance)):
            self.goal = goal.bounding_box
            self.goal_center = goal.center
        else:
            raise ValueError(f"Invalid goal {goal}")

    @property
    def approaching_actions(self):
        original_distance = np.linalg.norm(self.center - self.goal_center)
        dynamic_center = self.center + MOVEMENTS_2D[1:]
        new_distance = np.linalg.norm(dynamic_center - self.goal_center, axis=1)
        return [
            i+1
            for i, d in enumerate(new_distance)
            if d < original_distance
        ]

    @property
    def reach_goal(self) -> bool:
        """
        Check if the nucleotide is completely within the area of the goal.
        """
        return all(
            [
                self.x_upper_left >= self.goal[0],
                self.y_upper_left >= self.goal[1],
                self.x_lower_right <= self.goal[2],
                self.y_lower_right <= self.goal[3]
            ]
        )
    
    @property
    def class_name(self):
        return self.name.split("_")[0]
    
    @property
    def distance_to_goal(self):
        # 曼哈顿距离
        return np.abs(self.center - self.goal_center).sum()
    
class dNTP(Droplet):
    def __init__(
            self,
            name: str,
            width: int,
            height: int,
            location: Union[Tuple[int, int, int, int], Tuple[int, int], List[int], np.ndarray], # (x_upper_left, y_upper_left)
            goal: Union[Tuple[int, int, int, int], List[int], np.ndarray], # (x_upper_left, y_upper_left, x_lower_right, y_lower_right)
            **kwargs
        ) -> None:
        super().__init__(name, width, height, location, goal, **kwargs)
    
    @property
    def content(self):
        return get_nucleotide_from_name(self.name)
    
    def __repr__(self) -> str:
        return f"{self.name} at {self.location} with size {self.width}x{self.height} routing to {self.goal}"


class ddNTP(Droplet):
    def __init__(
            self,
            name: str,
            width: int,
            height: int,
            location: Union[Tuple[int, int, int, int], Tuple[int, int], List[int], np.ndarray], # (x_upper_left, y_upper_left)
            goal: Union[Tuple[int, int, int, int], List[int], np.ndarray], # (x_upper_left, y_upper_left, x_lower_right, y_lower_right)
            **kwargs
        ) -> None:
        super().__init__(name, width, height, location, goal, **kwargs)
    
    @property
    def content(self):
        return get_nucleotide_from_name(self.name)

class Intermediate(Droplet):
    def __init__(
            self,
            name: str,
            width: int,
            height: int,
            location: Union[Tuple[int, int], List[int], np.ndarray], # (x_upper_left, y_upper_left)
            goal: Union[Tuple[int, int, int, int], List[int], np.ndarray], # (x_upper_left, y_upper_left, x_lower_right, y_lower_right)
            **kwargs
        ) -> None:
        super().__init__(name, width, height, location, goal, **kwargs)
        assert name[0] in "ATCG", f"Invalid intermediate name {name}"

    @property
    def content(self):
        return get_nucleotide_from_name(self.name)