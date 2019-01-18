#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author            : Aaron Niskin <aaron@niskin.org>
# Date              : 2019-01-16
# Last Modified Date: 2019-01-18
# Last Modified By  : Aaron Niskin <aaron@niskin.org>
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
# from copy import deepcopy

LOGGER = logging.getLogger(__file__)
SH = logging.StreamHandler()
SH.setLevel(logging.DEBUG)
LOGGER.addHandler(SH)

SAMPLE_BOARD = {
        '_0': 5,
        '_1': 3,
        '_4': 7,
        '_9': 6,
        '_12':1,
        '_13':9,
        '_14':5,
        '_19':9,
        '_20':8,
        '_25':6,
        '_27':8,
        '_31':6,
        '_35':3,
        '_36':4,
        '_39':8,
        '_41':3,
        '_44':1,
        '_45':7,
        '_49':2,
        '_53':6,
        '_55':6,
        '_60':2,
        '_61':8,
        '_66':4,
        '_67':1,
        '_68':9,
        '_71':5,
        '_76':8,
        '_79':7,
        '_80':9
        }

SBOARD2 = {
        '_7': 1,
        '_14':2,
        '_17':3,
        '_21':4,
        '_33':5,
        '_36':4,
        '_38':1,
        '_39':6,
        '_47':7,
        '_48':1,
        '_55':5,
        '_60':2,
        '_67':8,
        '_70':4,
        '_73':3,
        '_75':9,
        '_76':1
        }

SBOARD3 = {
        '_0':8,
        '_1':6,
        '_4':2,
        '_12':7,
        '_16':5,
        '_17':9,
        '_31':6,
        '_33':8,
        '_37':4,
        '_47':5,
        '_48':3,
        '_53':7,
        '_64':2,
        '_69':6,
        '_74':7,
        '_75':5,
        '_77':9
        }


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
        self.static = False
        self.dim = dim
        self.options = set(range(dim**2))

    def _is_static(self):
        """Get the sudoku board vector
        """
        return self.static

    def set_value(self, val):
        """Sets the value of the cell

        IDEMPOTENT

        args:
            val : int
                The value to remove.
        """
        assert (0 <= val < self.dim**2), "Innapropriate value"
        if val not in self.options:
            return False
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

    def is_solved(self):
        """Is this cell figured out?"""
        return self.val is not None

    def copy(self):
        """Copy the cell"""
        cell = SudokuCell(self.val, self.dim)
        cell.static = self.static
        cell.options = self.options
        return cell

    def __str__(self):
        """return a string representation of the cell (just the value)"""
        return '  % 3s  ' % (str(self.val+1) if self.val is not None else '')


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

    def __init__(self, dim, **kwargs):
        self.dim = dim
        self.board = [SudokuCell(id_, dim) for id_ in range(dim**4)]
        for id_, val in kwargs.items():
            id_ = int(id_.strip('_'))
            assert self.make_move(id_, val-1), 'Invalid starting position for a board'
            self.board[id_].static = True

    def copy(self):
        """make a copy of the board"""
        board = SudokuBoard(self.dim)
        board.board = [x.copy() for x in self.board]
        return board

    def make_move(self, id_, val):
        """make a move (and propagate the changes"""
        if not self.board[id_].set_value(val):
            return False
        for other_id in _id_to_related(id_, self.dim):
            other_val = self.board[other_id].remove_option(val)
            if other_val is not None:
                self.make_move(other_id, other_val)
        return True

    def __str__(self):
        """Get the sudoku board vector"""
        line_len = None
        outstr = '|'
        for id_, cell in enumerate(self.board):
            outstr += '|'+cell.__str__()
            if (id_ % self.dim) == (-1 % self.dim):
                outstr += '|'
                if (id_ % (self.dim**2)) == (-1 % (self.dim**2)):
                    outstr += '|'
                    if line_len is None:
                        line_len = len(outstr)
                        outstr = '-'*line_len + '\n' + outstr
                    outstr += '\n'
                    if (id_ % (self.dim**3)) == (-1 % (self.dim**3)):
                        outstr += '='*line_len + '\n'
                    else:
                        outstr += '-'*line_len + '\n'
                    outstr += '|'
        return " Sudoku board:\n" + ('='*line_len) + '\n' + outstr

    def print_board(self):
        """print board"""
        print(self.__str__())

    def __itercols__(self):
        for i in range(self.dim**2):
            yield _col_to_ids(i, self.dim)

    def __iterrows__(self):
        for i in range(self.dim**2):
            yield _row_to_ids(i, self.dim)

    def __iterboxes__(self):
        for i in range(self.dim**2):
            yield _box_to_ids(i, self.dim)

    def __step_1_inner__(self, items):
        changed = False
        for i in range(self.dim**2):
            cells = [x for x in items if i in self.board[x].options]
            if len(cells) == 1:
                self.make_move(cells[0], i)
                changed = True
        return changed

    def __step_1__(self):
        changed = False
        for row in self.__iterrows__():
            changed = changed or self.__step_1_inner__(row)
        for col in self.__itercols__():
            changed = changed or self.__step_1_inner__(col)
        for box in self.__iterboxes__():
            changed = changed or self.__step_1_inner__(box)
        return changed

    def solve(self):
        """Solve this board!"""
        changed = True
        while changed:
            changed = False
            for id_, cell in enumerate(self.board):
                if len(cell.options) == 1:
                    val = list(cell.options)[0]
                    self.make_move(id_, val)
                    changed = True
            changed = changed or self.__step_1__()
    def __getitem__(self, i):
        return self.board[i]

def test_functions(func, dim=3, indx=0):
    """docstring"""
    brd = SudokuBoard(dim)
    for id_ in func(indx, dim):
        brd[id_].val = 'H'
    brd[indx].val = 'O'
    return brd