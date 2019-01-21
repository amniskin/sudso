SUDoku-SOlver
-------------

Various approaches to sudoku solving. NOTE: This is a python3 library, unfortunately.

To install, run the following from a terminal:
	>>> pip install .


For now, to run it like so (but this will change eventually):

	>>> import sudso
	>>> from sudso.solvers import PidgeonHoleBoard
	>>> base_board = sudso.SudokuBoard.from_json(sudso.SBOARD1)  # creates 9 by 9 sudoku board
	>>> print(base_board)  # observe the board in all its complexity
	>>> board = PidgeonHoleBoard.from_board(base_board)
	>>> board.solve()
	>>> print(board)  # observe the glory!


It seems to be working so far... We need more boards.
