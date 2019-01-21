#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author            : Aaron Niskin <aaron@niskin.org>
# Date              : 2019-01-16
# Last Modified Date: 2019-01-20
# Last Modified By  : Aaron Niskin <aaron@niskin.org>
# pylint: disable=fixme
"""
Build the sudoku board object, along with all appropriate methods.

The board IDs will be arranged thusly (for simplicity, dim=3 in this example):

-------------------------------
| column || 0  | 1  | 2  | 3  |
|==============================
| row  0 || 0  | 1  | 2  | 3  |
-------------------------------
|      1 || 4  | 5  | 6  | 7  |
-------------------------------
|      2 || 8  | 9  | 10 | 11 |
-------------------------------
|      3 || 12 | 13 | 14 | 15 |
-------------------------------


The basic structure is:
    cells keep track of everything pertaining to that individual cell. e.g.
        * It's own value -- if it hsa one
        * set of possible values
        * color -- for printing reasons
        * failed guesses
"""
import logging
from itertools import combinations
from functools import reduce
from copy import copy
from math import ceil
from sudso.core import SudokuBoard, SudokuCell

LOGGER = logging.getLogger(__file__)
SH = logging.StreamHandler()
SH.setLevel(logging.DEBUG)
LOGGER.addHandler(SH)


class PidgeonHoleCell(SudokuCell):
    """The sudoku board cell class

    cells keep track of everything pertaining to that individual cell. e.g.
        * It's own value -- if it hsa one
        * set of possible values

    args:
        id_ : int
            The cell ID
        dim : int
            "dimension of the board". For instance, a traditional sudoku board
            would have dim == 3. Note: the actual number of squares in a sudoku
            board are dim^4 not dim^2.
    """

    def __init__(self, id_=0, dim=3, options=None, val=None, color='34',
                 good_color='34', bad_color='35'):
        # pylint: disable=too-many-arguments
        super().__init__(val=None, color='34',
                         good_color='34', bad_color='35')
        self.id_ = id_
        self.dim = dim
        if options is None:
            self.options = set(range(1, dim**2+1))  # all possible options
        else:
            self.options = options

    def set_value(self, val):
        """Sets the value of the cell

        IDEMPOTENT

        args:
            val : int
                The value to remove.
        """
        assert (val in self.options) or (val == self.get_value()),\
               "Innapropriate value in cell %i" % self.id_
        super().set_value(val)
        self.options.clear()
        return True

    def remove_option(self, val):
        """Remove an option from the list of options

        IDEMPOTENT

        args:
            val : int
                The value to remove.
        """
        if val in self.options:
            self.options.remove(val)
        if len(self.options) == 1:
            return list(self.options)[0]
        return None

    def get_options(self):
        """get all available options"""
        return self.options


class PidgeonHoleBoard(SudokuBoard):
    """The sudoku board class

    args:
        dim : int
            "dimension of the board". For instance, a traditional sudoku board
            would have dim == 3. Note: the actual number of squares in a sudoku
            board are dim^4 not dim^2.
        kwargs : keys = int, vals = int
            the values are the values to initialize the board with, and the
            keys are the IDs of the cells in question.
    """
    CELL_TYPE = PidgeonHoleCell
    def __init__(self, dim=3, board=None, boards=None, good_color='34', bad_color='35'):
        # pylint: disable=redefined-outer-name,too-many-arguments
        if board is None:
            board = [self.CELL_TYPE(id_=id_, dim=dim, good_color=good_color,
                                    bad_color=bad_color) for id_ in range(dim**4)]
        super().__init__(board=board, dim=dim)
        if boards is None:
            self.boards = []
        else:
            self.boards = boards

    def make_move(self, id_, val):
        """make a move (and propagate the changes"""
        valid_move = super().make_move(id_, val)
        assert valid_move, 'Cell %i recieved invalid value %i' % (id_, val)
        for other_id in self.get_relatives(id_, relationship='all'):
            self.remove_option(other_id, val)
        return True

    def remove_option(self, id_, val):
        """remove an option from a cell"""
        opt_val = self[id_].remove_option(val)
        if opt_val is not None:
            return self.make_move(id_, opt_val)
        return False

    def make_guess(self):
        """make a guess"""
        this_cell = None
        for cell in self:
            guesses = cell.get_options()
            if guesses:
                this_cell = cell
                this_guess = list(guesses)[0]
                break
        if this_cell is None:
            return False
        self.boards.append(((this_cell.id_, this_guess),
                            self.copy_board()))
        self.make_move(this_cell.id_, this_guess)
        return True

    def revert_guess(self):
        """revert a guess"""
        if not self.boards:
            return False
        (cell, guess), old_board = self.boards.pop(-1)
        self.board = old_board
        self.remove_option(cell, guess)  # guess didn't work
        return True

    def pidgeonhole(self, num):
        """The pidgeonhole principle part of this thing"""
        def inner_func(items):
            inner_changed = False
            if num < 1:
                return inner_changed
            for vals in map(set, combinations(range(1, self.dim**2+1), num)):
                # if there is a set of N values with only N cells that can be
                # them, then pidgeon hole principle tells us that they each
                # must be one of those N values. So we can remove all other
                # values from these N
                cells = {x for x in items if vals.intersection(self[x].options)}
                # to make sure it isn't really 5 cells only representing 4
                # values (because one never shows up)
                all_vals_represented = vals.issubset(reduce(lambda x, y: x.union(self[y].options),
                                                            cells, set()))
                if len(cells) == num and all_vals_represented:
                    # remove all other options from these cells
                    for cell in cells:
                        for k in self[cell].options.difference(vals):
                            newval = self.remove_option(cell, k)
                            inner_changed = inner_changed or newval
                # if there is a set of N cells that can only be N values,
                # similar logic applies -- but in this case, we know no other
                # cells can be these values.
                cells = {x for x in items if self[x].options.issubset(vals) and self[x].options}
                all_vals_represented = vals.issubset(reduce(lambda x, y: x.union(self[y].options),
                                                            cells, set()))
                if len(cells) == num and all_vals_represented:
                    # remove these options from all other cells
                    for cell in [x for x in items if x not in cells]:
                        for k in vals:
                            newval = self.remove_option(cell, k)
                            inner_changed = inner_changed or newval
            return inner_changed
        changed = False
        for items in self.__iterneighbors__():
            changed = inner_func(items) or changed
        return changed

    def solve(self):
        """Solve this board!"""
        # pylint: disable=redefined-outer-name
        changed = True
        while changed:
            changed = False
            for i in range(2, ceil(self.dim**2/2)):
                for j in range(1, i):
                    changed = self.pidgeonhole(j) or changed
                    if not self.is_valid():
                        if self.boards:
                            changed = self.revert_guess() or changed
                        else:
                            return False
                    elif self.complete():
                        return True
            if not changed:
                changed = self.make_guess()
        return False  # if it was never completed


if __name__ == '__main__':
    from sudso.example_boards import BOARDS
    for b in BOARDS:
        print('================================')
        board0 = copy(SudokuBoard.from_json(b))
        board = copy(PidgeonHoleBoard.from_board(board0))
        board.solve()
        print(board)
        print(board.is_valid(), board.complete())

    print('random boards:')
    for i in range(401):
        try:
            board0 = SudokuBoard.get_random_board(3)
            board = PidgeonHoleBoard.from_board(board0)
            print('num iters: ', i)
            print(board0)
            board.solve()
            print(board)
            print(board.is_valid(), board.complete())
            break
        except AssertionError:
            pass
