NAME          MIP_VARIED_EX01
ROWS
 N   OBJ
 E   C1
 L   C2
COLUMNS
    X1       OBJ       4
    X1       C1       3
    X2       OBJ       4
    X3       OBJ       5
    X3       C1       3
RHS
    RHS1     C1        3
    RHS1     C2        3
BOUNDS
 LI BND1     X1         0      ! X1 ≥ 0 (整数下界)
 UI BND1     X1        10      ! X1 ≤ 10 (整数上界)
 LI BND1     X2         0      ! X2 ≥ 0
 UI BND1     X2         5      ! X2 ≤ 5
 LI BND1     X3         0      ! X3 ≥ 0
 UI BND1     X3         8      ! X3 ≤ 8
ENDATA