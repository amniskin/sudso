# SUDoku-SOlver
Various approaches to sudoku solving algorithms.


For now, to run it:

```python
import sudoku_solver as sudso

board = sudso.SudokuBoard(3, **sudso.SBOARD1)  # creates 9 by 9 sudoku board
board.print_board()  # observe the board in all its complexity

board.solve()
board.print_board()  # observe the glory!
```


Or:


```python
import sudoku_solver as sudso

board = sudso.SudokuBoard(3, **sudso.SBOARD2)  # creates 9 by 9 sudoku board
board.print_board()  # observe the board in all its complexity

board.solve()
board.print_board()  # observe the shame!
```
