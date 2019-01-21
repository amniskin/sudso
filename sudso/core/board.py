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
import random
from copy import copy, deepcopy
from itertools import combinations

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
    """

    def __init__(self, val=None, color='34', good_color='34',
                 bad_color='35'):
        # pylint: disable=too-many-arguments
        self.val = val
        self.color = color
        self.good_color = good_color
        self.bad_color = bad_color  # for identifying erroneous moves easily

    def set_value(self, val):
        """Sets the value of the cell

        IDEMPOTENT

        args:
            val : int
                The value to remove.
        """
        self.val = val
        return True

    def get_value(self):
        """getter"""
        return self.val

    def bad_cell(self):
        """designate this cell as bad"""
        self.color = self.bad_color

    def good_cell(self):
        """Designate this cell as 'good'"""
        self.color = self.good_color

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
    CELL_TYPE = SudokuCell
    def __init__(self, dim=3, board=None, good_color='34', bad_color='35'):
        # pylint: disable=redefined-outer-name
        self.dim = dim
        if board is None:
            self.board = [self.CELL_TYPE(good_color=good_color,
                                         bad_color=bad_color) for _ in range(dim**4)]
        else:
            self.board = [self.CELL_TYPE(**deepcopy(x.__dict__)) for x in board]
            assert len(board) == self.dim**4, 'Invalid board size'
            self.board = board

    def _set_value(self, id_, val):
        """set the value of a cell explicitly"""
        return self[id_].set_value(val)

    def make_move(self, id_, val):
        """set the value of a cell -- can be replaced with more complicated
        logic later and things should still work
        """
        return self._set_value(id_, val)

    def copy_board(self):
        """make a copy of the board"""
        return [deepcopy(x) for x in self]

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
        return " \n" + ('='*line_len) + '\n' + outstr + '|'*(line_len - 1)

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

    def get_relatives(self, id_, relationship='all'):
        """get all ids in the same {box,row,column}

        args:
            id_ : int
                the cell ID you want the relatives of
            relationship : str
                {box,row,col,all}
        """
        if relationship == 'all':
            return _id_to_related(id_, self.dim)
        if relationship == 'box':
            return _id_to_box_id(id_, self.dim)
        if relationship == 'row':
            return _id_to_row(id_, self.dim)
        if relationship == 'col':
            return _id_to_col(id_, self.dim)
        return False

    def good_cell(self, id_):
        """designate a cell as bad"""
        return self[id_].good_cell()

    def bad_cell(self, id_):
        """designate a cell as bad"""
        return self[id_].bad_cell()

    @classmethod
    def from_json(cls, js_dict):
        """convert json to board"""
        # pylint: disable=redefined-outer-name,invalid-name
        dim = js_dict.pop('dim')
        board = cls(dim=dim)
        for k, v in js_dict.items():
            id_ = int(k.strip('_'))
            board.make_move(id_, v)
        return board

    def is_valid(self):
        """docstring"""
        # pylint: disable=invalid-name
        for items in self.__iterneighbors__():
            item_vals = [self[x].val for x in items if self[x].val is not None]
            if len(set(item_vals)) != len(item_vals):
                dups = [(x, y) for x, y in combinations(items, 2) if self[x].val == self[y].val]
                for x, y in dups:
                    self.bad_cell(x)
                    self.bad_cell(y)
                return False
        return True

    def complete(self):
        """Check if the board is completed without errors"""
        for cell in self:
            if cell.get_value() is None:
                return False
        return self.is_valid()

    def __getitem__(self, i):
        return self.board[i]

    def __iter__(self):
        return iter(self.board)

    @classmethod
    def from_board(cls, sudokuboard_obj):
        """Copy a board into another
        This is particularly useful for solver classes that inherit from this
        one... They can take a SudokuBoard object, and import it into their own
        class and do whatever to it. The thing calls "make_move" on each cell
        which a value that is not None

        args:
            board : SudokuBoard object
        returns:
            Whatever class this is initialized with the values from this
            SudokuBoard object
        """
        in_dict = deepcopy(sudokuboard_obj.__dict__)
        inboard = in_dict.pop('board')
        outboard = cls(**in_dict)
        for id_, cell in enumerate(inboard):
            value = cell.get_value()
            if value is not None:
                outboard.make_move(id_, value)
        return outboard

    @classmethod
    def get_random_board(cls, dim, pct=0.05, seed=1234):
        """Generate a random sudoku board"""
        random.seed(seed)
        outboard = cls(dim=dim)
        num_cells = len(outboard.board)
        sample = random.sample(range(num_cells), int(num_cells*pct))
        if len(sample) != len(set(sample)):
            print('ruh roh')
        for cell in sample:
            outboard.make_move(cell, random.randint(1, dim**2))
        return outboard


if __name__ == '__main__':
    import sys
    import os
    LIBPATH = os.path.normpath(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                            os.pardir, os.pardir))
    sys.path.append(LIBPATH)
    from sudso.example_boards import BOARDS
    for b in BOARDS:
        print('================================')
        board = copy(SudokuBoard.from_json(b))
        print(board)
        print(board.is_valid(), board.complete())
