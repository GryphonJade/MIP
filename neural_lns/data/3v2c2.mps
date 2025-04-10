NAME          MIP_EX7
ROWS
 N  OBJ
 E  C1
 L  C2
COLUMNS
    X1      OBJ     1
    X1      C1      1
    X2      OBJ     2
    X2      C1      1
    X2      C2      1
    X3      OBJ     3
    X3      C2      1
RHS
    RHS1    C1      3
    RHS1    C2      4
BOUNDS
 LI BND1    X1     0
 UI BND1    X1    10
 LI BND1    X2     0
 UI BND1    X2    10
 LI BND1    X3     0
 UI BND1    X3    10
INTORG
    MARKER  'MARKER' 'INTORG'
    X1      '        '
INTEND
    MARKER  'MARKER' 'INTEND'
ENDATA