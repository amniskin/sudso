# SUDoku-SOlver
Various approaches to sudoku solving algorithms.


For now, to run it like so (but this will change eventually):

```python
import sudoku_board as sudso

board = sudso.SudokuBoard(3, **sudso.SBOARD1)  # creates 9 by 9 sudoku board
print(board)  # observe the board in all its complexity

board.solve()
print(board)  # observe the glory!
```


Or:


```python
import sudoku_board as sudso

board = sudso.SudokuBoard(3, **sudso.SBOARD2)  # creates 9 by 9 sudoku board
print(board)  # observe the board in all its complexity

board.solve()
print(board)  # observe the shame!
```
