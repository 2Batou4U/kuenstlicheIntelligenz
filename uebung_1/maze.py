from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import randint
import enum
# Try to import the numba just-in-time compiler
# This is optional and will not result in an error but will speed up the maze generation significantly!
try:
    from numba import njit
except ImportError:
    # use a dummy wrapper instead
    # this will be significantly slower
    def njit(*args, **kwargs):
        def __wrapped__(function):
            return function
        return __wrapped__   
    
import numpy.typing as npt
from typing import Tuple, Optional, Any, Set


class MazeTiles(enum.IntEnum):
    EMPTY: int = 0  # value of an empty tile
    WALL: int = 5   # value of a wall tile
    START: int = 1  # value of the start tile. This is only used when writing the maze to an ASCII representation.
    END_: int = 2    # value of the goal tile. This is only used when writing the maze to an ASCII representation.
    
    
# numba doesn't like Enums so we move the required values out out of the class
_WALL: int = MazeTiles.WALL
_EMPTY: int = MazeTiles.EMPTY
    
    
# optimized function to fill the labyrinth with walls
@njit()
def _create_islands(shape: Tuple[int, int], 
                   Z: npt.NDArray[np.int64], 
                   P: npt.NDArray[int], 
                   n_density: int, 
                   n_complexity: int, 
                   seed: Optional[Any] = None) -> None:
    if seed is not None:
        np.random.seed(seed)
    # Create islands
    for i in range(n_density):
        # Test for early stop: if all starting point are busy, this means we
        # won't be able to connect any island, so we stop.
        T = Z[2:-2:2, 2:-2:2]
        if T.sum() == T.size:
            break

        x, y = P[i]
        Z[y, x] = _WALL
        for j in range(n_complexity):
            neighbours = []
            if x > 1:
                neighbours.append([(y, x - 1), (y, x - 2)])
            if x < shape[1] - 2:
                neighbours.append([(y, x + 1), (y, x + 2)])
            if y > 1:
                neighbours.append([(y - 1, x), (y - 2, x)])
            if y < shape[0] - 2:
                neighbours.append([(y + 1, x), (y + 2, x)])
            if len(neighbours):
                choice = np.random.randint(len(neighbours))
                next_1, next_2 = neighbours[choice]
                if Z[next_2] == _EMPTY:
                    Z[next_1] = Z[next_2] = _WALL
                    y, x = next_2
            else:
                break
    

class Maze(object):
    """
    Class representing a Maze. By default it is randomly generated, but can be initialized from a file with 
    'load_from_file' as well.
    """
    
    MAZE_DTYPE: np.dtype = np.int8  # data type of the maze representation
       
    # color values for the plot
    START_COLOR: str = 'blue'
    END_COLOR: str = 'red'

    def __init__(self, 
                 shape: Tuple[int, int], 
                 complexity: float = 0.75,
                 density: float = 0.50, 
                 seed: Optional[Any] = None, 
                 _generate: bool = True):
        """ 
        Initialize a maze with the given shape.
        Optionally, a maze is generated procedurally using the randomized prim's algorithm
        (https://en.wikipedia.org/wiki/Maze_generation_algorithm).
        
        Parameters:
        -----------
        shape:
            The shape of the maze to create.
            Even values will be increased by one so the resulting shape is always odd.
        complexity:
            The complexity parameter when randomly generated (between 0.0 and 1.0).
        density:
            The density of the randomly generated maze (between 0.0 and 1.0). Low values will
            result in more empty space between the walls of the maze.
        seed:
            Seed to use for the random generation. By default it is initialized with the system's entropy
        _generate:
            Generate a random maze. Private parameter, do not change!
        """

        # Only odd shapes are allowed
        self.shape = ((shape[0] // 2) * 2 + 1, (shape[1] // 2) * 2 + 1)
        self.board = np.zeros(self.shape, dtype=self.MAZE_DTYPE)
        
        # Fill borders
        self.board[0, :] = self.board[-1, :] = self.board[:, 0] = self.board[:, -1] = MazeTiles.WALL
        self.start = (1,1)
        self.end = (self.shape[0]-2, self.shape[1]-2)
        if _generate:
            # generate the rest of the maze
            self.__create(self.shape, complexity, density, seed)

    def to_ascii(self) -> str:
        """
        Convert the maze into an ascii representation that can be printed on the command line or stored in a file.
        
        Returns:
        --------
        rep:
            An ASCII representation of the maze.
        """
        rep = ''

        mapping = {MazeTiles.WALL: 'X',
                   MazeTiles.EMPTY: ' ',
                   MazeTiles.START: 'S',
                   MazeTiles.END: 'G'}
        board_copy = self.board.copy()
        board_copy[self.start] = MazeTiles.START
        board_copy[self.end] = MazeTiles.END
        for i in range(self.nrows):
            for j in range(self.columns):
                try:
                    rep += mapping[board_copy[i, j]]
                except KeyError:
                    rep += ' '
            rep += '\n'
        return rep

    def plot(self, fig: Optional[plt.Figure] = None) -> plt.Figure:
        """
        Plot the maze. This requires the matplotlib library.
        
        Returns:
        --------
        figure:
            The figure of the resulting plot.
        """
        
        if fig is None:
            fig = plt.figure(figsize=(10, 5))
        p = plt.imshow(self.board, cmap=plt.cm.binary, interpolation='nearest', zorder=-1)
        plt.scatter(self.start[1], self.start[0], c=self.START_COLOR, zorder=5)
        plt.scatter(self.end[1], self.end[0], c=self.END_COLOR, zorder=5)
        plt.gca().set_xticks(np.arange(-.5, self.shape[1], 1), minor=True)
        plt.gca().set_xticks(np.arange(self.shape[1], 5))
        plt.gca().set_yticks(np.arange(-.5, self.shape[0]+0.5, 1), minor=True)
        plt.gca().set_yticks(np.arange(self.shape[0], 5))
        plt.grid(which='minor')
        return fig

    def write_to_file(self, filename: str) -> None:
        """
        Write the maze into the given file in ASCII representation.
        
        Parameters:
        -----------
        filename:
            The file to write
        """
        f = open(filename, 'w')
        f.write(self.to_ascii())
        f.close()

    @property
    def nrows(self) -> int:
        """
        Property to return the number of rows of the maze.
        """
        return self.shape[0]

    @property
    def ncolumns(self) -> int:
        """
        Property to return the number of columns of the maze.
        """
        return self.shape[1]

    def is_wall(self, x: int, y: int) -> bool:
        """
        Return True if the given coordinate is a wall tile
        
        Parameters:
        -----------
        x:
            the x coordinate
        y:
            the y coordinate
            
        Returns:
        --------
            True if a wall tile, False otherwise
        """
        return self.board[x, y] == MazeTiles.WALL

    def is_empty(self, x: int, y: int) -> bool:
        """
        Return True if the given coordinate is an empty tile.
        
        Parameters:
        -----------
        x:
            the x coordinate
        y:
            the y coordinate
            
        Returns:
        --------
            True if an empty tile, False otherwise
        """
        return not self.board[x, y] == MazeTiles.EMPTY

    def __get_dead_ends(self) -> npt.NDArray[int]:
        """
        Internal function. DO NOT USE
        """
        from scipy.signal import convolve2d
        board = self.board
        kernel = np.ones((3,3), dtype=self.MAZE_DTYPE)
        kernel[1,1] = 0

        indices = np.where(np.logical_and(board == 0, convolve2d(board, kernel, fillvalue=1, mode='same') == 7))
        return indices

    def set_start_end_random(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        Set the start and end of the maze to random locations.
        Both coordinates will always be placed in dead ends of the maze.
        
        Returns:
        --------
        start:
           the starting location
        end:
            the end location
        """
        dead_ends = self.__get_dead_ends()
        indices = np.arange(0, dead_ends[0].shape[0])
        indices = np.random.choice(indices, 2)
        self.start = (dead_ends[0][indices[0]], dead_ends[1][indices[0]])
        self.end = (dead_ends[0][indices[1]], dead_ends[1][indices[1]])

        return self.start, self.end

    def set_start_end(self, start: Tuple[int, int], end: Tuple[int, int]):
        """
        Set the start and end of the maze to specifiy values. 
        If one of the specified tiles is a WALL, an exception will be raised.
        
        Parameters:
        -----------
        start: 
            The start coordinate
        end:
            The end coordinate
        """
        if any(self.is_wall(*c) for c in (start, end)):
               raise ValueError("Specified coordinate is a wall")
               
        self.start = tuple(start)
        self.end = tuple(end)

    def get_neighbors(self, 
                      coord: Tuple[int, int], 
                      visited: Set[Tuple[int, int]] = set()) -> List[Tuple[int, int]]:
        """
        Get all neighbors of a given coordinate. The order will be always left, right, up, down.
        
        Parameters:
        -----------
        coord:
            The coordinate as tuple of x and y
        visited: 
            Optional visited set to exclude neighbors that have already been visited
            
        Returns:
        --------
        List of all neighbors as (x,y) tuples            
        """
        neighbors = [(coord[0] - 1, coord[1]), (coord[0] + 1, coord[1]), (coord[0], coord[1] - 1),
                    (coord[0], coord[1] + 1)]
        return [c for c in neighbors if self.board[c[0], c[1]] != MazeTiles.WALL and c not in visited]

    def get_neighbors_iter(self, coord: Tuple[int, int], visited: Set[Tuple[int, int]] = set()):
        """
        Returns the neighbors as an iterator.
        See docstring of get_neighbors for an explanation of the parameters.
        """
        c = (coord[0], coord[1] + 1)
        if self.board[c[0], c[1]] != MazeTiles.WALL and c not in visited:
            yield c
        c = (coord[0], coord[1] - 1)
        if self.board[c[0], c[1]] != MazeTiles.WALL and c not in visited:
            yield c
        c = (coord[0] + 1, coord[1])
        if self.board[c[0], c[1]] != MazeTiles.WALL and c not in visited:
            yield c
        c = (coord[0] - 1, coord[1])
        if self.board[c[0], c[1]] != MazeTiles.WALL and c not in visited:
            yield c

    @classmethod
    def load_from_file(cls, filename: str) -> Maze:
        """
        Load a new maze from a file.
        
        Parameters:
        -----------
        filename: 
            The file to load
            
        Returns:
        --------
            New maze loaded from file
        """
        with open(filename, 'r') as f:
            content = f.readlines()

        # remove whitespace characters like `\n` at the end of each line
        content = [x.strip() for x in content]

        inverse_mapping = {'X': MazeTiles.WALL,
                           ' ': MazeTiles.EMPTY}
        xss = []
        start = None
        end = None
        for i, line in enumerate(content):
            xs = []
            for j, c in enumerate(line):
                try:
                    xs.append(inverse_mapping[c])
                except KeyError as e:
                    xs.append(MazeTiles.EMPTY)
                    if c == 'S':
                        start = (i, j)
                    elif c == 'G':
                        end = (i, j)
                    else:
                        raise e

            xss.append(xs)

        if (start is None) ^ (end is None):
            raise ArithmeticError("Either start or end not defined")

        maze = Maze((len(xss), len(xss[0])), _generate=False)
        if start is not None:
            maze.set_start_end(start, end)
        for xs in xss:
            assert len(xs) == maze.ncolumns

        maze.board = np.asarray(xss)
        return maze

    def __create(self, 
                 shape: Tuple[int, int], 
                 complexity: float = 0.75,
                 density: float = 0.50, 
                 seed: Optional[Any] = None) -> npt.NDArray[int]:
        """
        Build a maze using given complexity and density. INTERNAL FUNCTION, DO NOT USE!
        For parameter explanation see the constructor.
        """
        if seed is not None:
            np.random.seed(seed)

        # Only odd shapes
        shape = ((shape[0] // 2) * 2 + 1, (shape[1] // 2) * 2 + 1)

        # Adjust complexity and density relatively to maze size
        n_complexity = int(complexity * (shape[0] + shape[1]))
        n_density = int(density * (shape[0] * shape[1]))

        # reinit actual maze
        self.board = np.zeros(shape, dtype=self.MAZE_DTYPE)
        Z = self.board

        # Fill borders
        Z[0, :] = Z[-1, :] = Z[:, 0] = Z[:, -1] = MazeTiles.WALL

        # Islands starting point with a bias in favor of border
        P = np.random.normal(0, 0.5, (n_density, 2))
        P = 0.5 - np.maximum(-0.5, np.minimum(P, +0.5))
        P = (P * [shape[1], shape[0]]).astype(int)
        P = 2 * (P // 2)

        WALL = MazeTiles.WALL
        EMPTY = MazeTiles.EMPTY

        _create_islands(shape, Z, P, n_density, n_complexity, seed=seed)
        return Z