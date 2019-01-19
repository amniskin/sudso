# SUDoku-SOlver
Various approaches to sudoku solving algorithms.


For now, to run it like so (but this will change eventually):

```python
import sudso

board = sudso.SudokuBoard(3, **sudso.SBOARD1)  # creates 9 by 9 sudoku board
board  # observe the board in all its complexity

board.solve()
board  # observe the glory!
```


It seems to be working so far... We need more boards.
