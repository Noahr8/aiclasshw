print_state([A,B,C,D,E,F,G,H,I]) :- maplist(write, [A,B,C,"\n",D,E,F,"\n",G,H,I]).

seq([X,Y,Z,_,_,_,_,_,_], [X,Y,Z], [0,1,2]).
seq([_,_,_,X,Y,Z,_,_,_], [X,Y,Z], [3,4,5]).
seq([_,_,_,_,_,_,X,Y,Z], [X,Y,Z], [6,7,8]).
seq([X,_,_,Y,_,_,Z,_,_], [X,Y,Z], [0,3,6]).
seq([_,X,_,_,Y,_,_,Z,_], [X,Y,Z], [1,4,7]).
seq([_,_,X,_,_,Y,_,_,Z], [X,Y,Z], [2,5,8]).
seq([X,_,_,_,Y,_,_,_,Z], [X,Y,Z], [0,4,8]).
seq([_,_,X,_,Y,_,Z,_,_], [X,Y,Z], [2,4,6]).

% valid state
x_o_diff([],0).
x_o_diff([x|T],Count) :- x_o_diff(T,CountPrev), Count is CountPrev + 1.
x_o_diff([o|T],Count) :- x_o_diff(T,CountPrev), Count is CountPrev - 1.
x_o_diff([b|T],Count) :- x_o_diff(T,Count).

valid_state(S) :- valid_elems(S), x_o_diff(S,0), \+ in_row_3(S).
valid_elems([A,B,C,D,E,F,G,H,I]) :- valid_elem(A), valid_elem(B), valid_elem(C), valid_elem(D), valid_elem(E),
    valid_elem(F), valid_elem(G), valid_elem(H), valid_elem(I).
in_row_3(S) :- seq(S, [X,Y,Z], _), X=o, Y=o,Z=o.
in_row_3(S) :- seq(S, [X,Y,Z], _), X=x, Y=x,Z=x.

valid_elem(E) :- E=o.
valid_elem(E) :- E=b.
valid_elem(E) :- E=x.

% Noah Robertson
% Start Work

% Approach:

% Create logic to solve a row
% Create logic to solve a column
% Create logic to solve a diagonal
% Create logic that two solutions are not the same
% Then two_ways checks all of these twice to see if there is two solutions.

% logic for rows:
% row_is_winnable if 
% seq(S, [X,Y,Z], [0,1,2])
% seq(S, [X,Y,Z], [3,4,5])
% seq(S, [X,Y,Z], [6,7,8])
% has 2 xs and 1 b

row_is_winnable(S) :- seq(S, [X,Y,Z], [0,1,2]), X=x, Y=x, Z=b.
row_is_winnable(S) :- seq(S, [X,Y,Z], [0,1,2]), X=x, Y=b, Z=x.
row_is_winnable(S) :- seq(S, [X,Y,Z], [0,1,2]), X=b, Y=x, Z=x.
row_is_winnable(S) :- seq(S, [X,Y,Z], [3,4,5]), X=x, Y=x, Z=b.
row_is_winnable(S) :- seq(S, [X,Y,Z], [3,4,5]), X=x, Y=b, Z=x.
row_is_winnable(S) :- seq(S, [X,Y,Z], [3,4,5]), X=b, Y=x, Z=x.
row_is_winnable(S) :- seq(S, [X,Y,Z], [6,7,8]), X=x, Y=x, Z=b.
row_is_winnable(S) :- seq(S, [X,Y,Z], [6,7,8]), X=x, Y=b, Z=x.
row_is_winnable(S) :- seq(S, [X,Y,Z], [6,7,8]), X=b, Y=x, Z=x.

%--------------------------------------------------------------
% logic for columns:
% col_is_winnable if
% seq(S, [X,Y,Z], [0,3,6])
% seq(S, [X,Y,Z], [1.4.7])
% seq(S, [X,Y,Z], [2,5,8])
% has 2 xs and 1 b

col_is_winnable(S) :- seq(S, [X,Y,Z], [0,3,6]), X=x, Y=x, Z=b.
col_is_winnable(S) :- seq(S, [X,Y,Z], [0,3,6]), X=x, Y=b, Z=x.
col_is_winnable(S) :- seq(S, [X,Y,Z], [0,3,6]), X=b, Y=x, Z=x.
col_is_winnable(S) :- seq(S, [X,Y,Z], [1,4,7]), X=x, Y=x, Z=b.
col_is_winnable(S) :- seq(S, [X,Y,Z], [1,4,7]), X=x, Y=b, Z=x.
col_is_winnable(S) :- seq(S, [X,Y,Z], [1,4,7]), X=b, Y=x, Z=x.
col_is_winnable(S) :- seq(S, [X,Y,Z], [2,5,8]), X=x, Y=x, Z=b.
col_is_winnable(S) :- seq(S, [X,Y,Z], [2,5,8]), X=x, Y=b, Z=x.
col_is_winnable(S) :- seq(S, [X,Y,Z], [2,5,8]), X=b, Y=x, Z=x.

%--------------------------------------------------------------
% logic for diagonals:
% diag_is_winnable if
% seq(S, [X,Y,Z], [0,4,8])
% seq(S, [X,Y,Z], [2,4,6])
% has 2 xs and 1 b

diag_is_winnable(S) :- seq(S, [X,Y,Z], [0,4,8]), X=x, Y=x, Z=b.
diag_is_winnable(S) :- seq(S, [X,Y,Z], [0,4,8]), X=x, Y=b, Z=x.
diag_is_winnable(S) :- seq(S, [X,Y,Z], [0,4,8]), X=b, Y=x, Z=x.
diag_is_winnable(S) :- seq(S, [X,Y,Z], [2,4,6]), X=x, Y=x, Z=b.
diag_is_winnable(S) :- seq(S, [X,Y,Z], [2,4,6]), X=x, Y=b, Z=x.
diag_is_winnable(S) :- seq(S, [X,Y,Z], [2,4,6]), X=b, Y=x, Z=x.

%--------------------------------------------------------------
% logic for no same spot
% is_sam_solution if
% two_ways(S)
% seq(S, [X,Y,Z], [row])
% seq(S, [X,Y,Z], [col])
% seq(S, [X,Y,Z], [diag])
% does intersect b

% rows and columns
rc_is_same_solution(S) :- seq(S, [B,X,X], [0,1,2]), seq(S, [B,X,X], [0,3,6]), X=x, B=b.
rc_is_same_solution(S) :- seq(S, [B,X,X], [3,4,5]), seq(S, [X,B,X], [0,3,6]), X=x, B=b.
rc_is_same_solution(S) :- seq(S, [B,X,X], [6,7,8]), seq(S, [X,X,B], [0,3,6]), X=x, B=b.

rc_is_same_solution(S) :- seq(S, [X,B,X], [0,1,2]), seq(S, [B,X,X], [1,4,7]), X=x, B=b.
rc_is_same_solution(S) :- seq(S, [X,B,X], [3,4,5]), seq(S, [X,B,X], [1,4,7]), X=x, B=b.
rc_is_same_solution(S) :- seq(S, [X,B,X], [6,7,8]), seq(S, [X,X,B], [1,4,7]), X=x, B=b.

rc_is_same_solution(S) :- seq(S, [X,X,B], [0,1,2]), seq(S, [B,X,X], [2,5,8]), X=x, B=b.
rc_is_same_solution(S) :- seq(S, [X,X,B], [3,4,5]), seq(S, [X,B,X], [2,5,8]), X=x, B=b.
rc_is_same_solution(S) :- seq(S, [X,X,B], [6,7,8]), seq(S, [X,X,B], [2,5,8]), X=x, B=b.

% column & diagonals
cd_is_same_solution(S) :- seq(S, [B,X,X], [0,3,6]), seq(S, [B,X,X], [0,4,8]), X=x, B=b.
cd_is_same_solution(S) :- seq(S, [X,X,B], [2,5,8]), seq(S, [X,X,B], [0,4,8]), X=x, B=b.

cd_is_same_solution(S) :- seq(S, [X,X,B], [0,3,6]), seq(S, [X,X,B], [2,4,6]), X=x, B=b.
cd_is_same_solution(S) :- seq(S, [B,X,X], [2,5,8]), seq(S, [B,X,X], [2,4,6]), X=x, B=b.

% rows & diagonals

rd_is_same_solution(S) :- seq(S, [B,X,X], [0,1,2]), seq(S, [B,X,X], [0,4,8]), X=x, B=b.
rd_is_same_solution(S) :- seq(S, [X,X,B], [0,1,2]), seq(S, [B,X,X], [2,4,6]), X=x, B=b.

rd_is_same_solution(S) :- seq(S, [B,X,X], [6,7,8]), seq(S, [X,X,B], [2,4,6]), X=x, B=b.
rd_is_same_solution(S) :- seq(S, [X,X,B], [6,7,8]), seq(S, [X,X,B], [0,4,8]), X=x, B=b.


%--------------------------------------------------------------------------------------
% two_ways is true if
% row_is_winnable(x), col_is_winnable(x)
% row_is_winnable(x), diag_is_winnable(x)
% col_is_winnable(x), diag_is_winnable(x)


two_ways_x(S) :- valid_state(S), row_is_winnable(S), col_is_winnable(S), \+ rc_is_same_solution(S), print_state(S).
two_ways_x(S) :- valid_state(S), row_is_winnable(S), diag_is_winnable(S), \+ rd_is_same_solution(S), print_state(S).
two_ways_x(S) :- valid_state(S), col_is_winnable(S), diag_is_winnable(S), \+ cd_is_same_solution(S), print_state(S).

%--------------------------------------------------------------------------------------
% Extra credit
no_ways_x(S) :- valid_state(S), \+row_is_winnable(S), \+col_is_winnable(S), \+diag_is_winnable(S), print_state(S).
