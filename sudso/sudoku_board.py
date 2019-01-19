#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author            : Aaron Niskin <aaron@niskin.org>
# Date              : 2019-01-16
# Last Modified Date: 2019-01-19
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
        * move IDs that involve that cell
            this is so that you can undo things later
            It's stored as a dictionary -- keys are the move ID and vals are
            the elements removed from the set of possible values since the
            move. -- Here moves are considered any time the algo makes a guess,
            as is necessary in underdetermined boards.
"""

import logging
from itertools import combinations
from functools import reduce
from copy import copy, deepcopy

LOGGER = logging.getLogger(__file__)
SH = logging.StreamHandler()
SH.setLevel(logging.DEBUG)
LOGGER.addHandler(SH)


def _rowcol_to_id(row, column, dim):
    """Convert row,column to ID"""
    return row*(dim**2) + column

def _id_to_rowcol(id_, dim):
    """Convert id to row, col"""
    col = id_ % (dim ** 2)
    row = (id_ - col) // (dim ** 2)
    return row, col

def _row_to_ids(row, dim):
    """Convert a row number to the IDs of the cells in the same row"""
    return [_rowcol_to_id(row, i, dim) for i in range(dim**2)]

def _id_to_row(id_, dim):
    """Convert id to the IDs of the cells in the same row"""
    col = id_ % (dim ** 2)
    row = (id_ - col) // (dim ** 2)
    return _row_to_ids(row, dim)

def _col_to_ids(col, dim):
    """Convert a col number to the IDs of the cells in the same col"""
    return [_rowcol_to_id(i, col, dim) for i in range(dim**2)]

def _id_to_col(id_, dim):
    """Convert id to the IDs of the cells in the same col"""
    col = id_ % (dim ** 2)
    return _col_to_ids(col, dim)

def _id_to_box(id_, dim):
    """Convert id to box ID"""
    row = id_ // (dim ** 3)
    col = (id_ % (dim ** 2)) // dim
    return row * dim + col

def _box_to_ids(box, dim):
    row0 = box - (box % dim)
    col0 = (box % dim) * dim
    retval = []
    for row in range(row0, row0+dim):
        vals = [_rowcol_to_id(row, col, dim) for col in range(col0, col0+dim)]
        retval.extend(vals)
    return retval

def _id_to_box_id(id_, dim):
    """Convert id to box ID"""
    return _box_to_ids(_id_to_box(id_, dim), dim)

def _id_to_related(id_, dim):
    retset = set()
    retset.update(_id_to_row(id_, dim))
    retset.update(_id_to_col(id_, dim))
    retset.update(_id_to_box_id(id_, dim))
    return retset

def test_functions(func, dim=3, indx=0):
    """docstring"""
    brd = SudokuBoard(dim)
    for id_ in func(indx, dim):
        brd[id_].val = 'H'
    brd[indx].val = 'O'
    return brd


class SudokuCell():
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

    def __init__(self, id_, dim):
        self.id_ = id_
        self.val = None
        self.dim = dim
        self.color = '34'
        self.options = set(range(1, dim**2+1))
        self.guesses = set()

    def set_value(self, val):
        """Sets the value of the cell

        IDEMPOTENT

        args:
            val : int
                The value to remove.
        """
        assert (val in self.options) or (val == self.val), "Innapropriate value"
        self.val = val
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

    def possible_guesses(self):
        """make a guess"""
        return self.options.difference(self.guesses)

    def add_guess(self, guess):
        """add a guess that didn't work out"""
        assert guess in self.possible_guesses(), 'Invalid guess!'
        self.guesses.add(guess)
        self.options.remove(guess)
        return True

    def bad_cell(self):
        """designate this cell as bad"""
        self.color = '35'

    def __deepcopy__(self, foo):
        """Copy the cell"""
        # pylint: disable=blacklisted-name
        cell = SudokuCell(self.id_, self.dim)
        cell.val = self.val
        cell.color = self.color
        cell.options = deepcopy(self.options)
        cell.guesses = deepcopy(self.guesses)
        return cell

    copy = __deepcopy__


    def __str__(self):
        """return a string representation of the cell (just the value)"""
        return '  \x1b[%sm% 3s\x1b[0m  ' % (self.color,
                                            str(self.val) if self.val is not None else '')


class SudokuBoard():
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
    def __init__(self, dim, init=True, **kwargs):
        self.dim = dim
        self.board = [SudokuCell(id_, dim) for id_ in range(dim**4)]
        self.boards = []
        for id_, val in kwargs.items():
            id_ = int(id_.strip('_'))
            if init:
                self.make_move(id_, val)
            else:
                self[id_].val = val

    def make_move(self, id_, val):
        """make a move (and propagate the changes"""
        assert self[id_].set_value(val), 'Cell %i recieved invalid value %i' % (id_, val)
        for other_id in _id_to_related(id_, self.dim):
            self.remove_option(other_id, val)
        return True

    def remove_option(self, id_, val):
        """remove an option from a cell"""
        opt_val = self[id_].remove_option(val)
        if opt_val is not None:
            return self.make_move(id_, opt_val)
        return False

    def __solver__(self, num):
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
            changed = changed or inner_func(items)
        return changed

    def make_guess(self):
        """make a guess"""
        this_cell = None
        for cell in self.board:
            guesses = cell.possible_guesses()
            if guesses:
                this_cell = cell
                this_guess = list(guesses)[0]
                break
        if this_cell is None:
            return False
        self.boards.append(((this_cell.id_, this_guess), self.copy_board()))
        self.make_move(this_cell.id_, this_guess)
        return True

    def revert_guess(self):
        """revert a guess"""
        if not self.boards:
            return False
        (cell, guess), self.board = self.boards.pop(-1)
        self[cell].add_guess(guess)
        return True

    def solve(self):
        """Solve this board!"""
        # self.init()
        changed = True
        while changed:
            changed = False
            for i in range(2, 6):  # TODO: remove hard coded magic number
                changed = False
                for j in range(1, i):
                    this_chng = self.__solver__(j)
                    changed = changed or this_chng
                    if not self.is_valid():
                        if self.boards:
                            reverted = self.revert_guess()
                            changed = changed or reverted
                        else:
                            return False
                if self.complete():
                    return True
            if not changed:
                guessed = self.make_guess()
                changed = guessed

    def copy_board(self):
        """make a copy of the board"""
        return [deepcopy(x) for x in self.board]

    def __deepcopy__(self, something):
        """make a copy of the board object"""
        new_board = SudokuBoard(self.dim)
        new_board.board = [deepcopy(x) for x in self.board]
        new_board.boards = [deepcopy(b) for b in self.boards]
        return new_board

    copy = __deepcopy__

    def __str__(self):
        """Get the sudoku board vector"""
        line_len = 8*(self.dim**2)+self.dim+2
        outstr = '|'
        for id_, cell in enumerate(self.board):
            outstr += '|'+cell.__str__()
            if (id_ % self.dim) == (-1 % self.dim):
                outstr += '|'
                if (id_ % (self.dim**2)) == (-1 % (self.dim**2)):
                    outstr += '|\n'
                    if (id_ % (self.dim**3)) == (-1 % (self.dim**3)):
                        outstr += '='*line_len + '\n'
                    else:
                        outstr += '-'*line_len + '\n'
                    outstr += '|'
        return " Sudoku board:\n" + ('='*line_len) + '\n' + outstr + '|'*(line_len - 1)

    def __repr__(self):
        return self.__str__()

    def __itercols__(self):
        for i in range(self.dim**2):
            yield _col_to_ids(i, self.dim)

    def __iterrows__(self):
        for i in range(self.dim**2):
            yield _row_to_ids(i, self.dim)

    def __iterboxes__(self):
        for i in range(self.dim**2):
            yield _box_to_ids(i, self.dim)

    def __iterneighbors__(self):
        for func in [self.__iterboxes__, self.__iterrows__, self.__itercols__]:
            for items in func():
                yield items

    def init(self):
        """ initialize the cells """
        for id_, cell in enumerate(self.board):
            if cell.val is not None:
                assert self.make_move(cell.id_, cell.val), ('Invalid board state!'
                                                            ' ID: %i val: %i' % (id_, cell.val))

    def is_valid(self):
        """docstring"""
        # pylint: disable=invalid-name
        for items in self.__iterneighbors__():
            item_vals = [self[x].val for x in items if self[x].val is not None]
            if len(set(item_vals)) != len(item_vals):
                dups = [(x, y) for x, y in combinations(items, 2) if self[x].val == self[y].val]
                for x, y in dups:
                    self[x].bad_cell()
                    self[y].bad_cell()
                return False
        return True

    def complete(self):
        """Check if the board is completed without errors"""
        for cell in self.board:
            if cell.val is None:
                return False
        return self.is_valid()

    def __getitem__(self, i):
        return self.board[i]

    def to_dict(self):
        """convert board to dictionary"""
        return {('_%i' %i): c.val for i, c in enumerate(self.board) if c.val is not None}


if __name__ == '__main__':
    import sys
    import os
    LIBPATH = os.path.normpath(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                            os.pardir))
    print(LIBPATH)
    sys.path.append(LIBPATH)
    from sudso.example_boards import BOARDS
    for b in BOARDS:
        print('================================')
        board = copy(SudokuBoard(3, init=True, **b))
        board.solve()
        print(board)
        print(board.is_valid(), board.complete())
