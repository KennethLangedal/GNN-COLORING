# GNN-based greedy graph coloring

Build by calling ```make```.

For best performance, use:

```
export OMP_NUM_THREADS=#
export OMP_PLACES=cores
export OMP_PROC_BIND=spread
```

ID, SDA, and SD, use recursive DFS, so stack overflow can happen. Use ```ulimit -s #``` to increase the stack size (replace # by a large number).

Usage:
```
./coloring -g [path to graph] [options] tc1 tc2 ... tcn
```

Where tc**x** is a number of threads to run experiments for.

For example:
```
./coloring -g web-Google.mtx 1 4
```
This will run sequential and parallel experiments using 1 and 4 threads.
The output of the program without -v is a single line per heuristic on the form:
heuristic_#culberson #colors_used seconds_sequential seconds_tc1 seconds_tc2 ... seconds_tcn

The output from the example above could look something like this:
```
FF_0 44 0.048468 0.309738 0.095933 
LF_0 45 0.374679 0.408236 0.116546 
LLF_0 44 0.398347 0.436153 0.119487 
SL_0 44 0.556319 0.632674 0.186328 
SLL_0 44 0.463689 0.577793 0.172596 
GNN2_0 44 0.538099 0.591235 0.162645 
GNN3_0 44 0.644065 0.688535 0.190566 
GNN4_0 44 0.756040 0.799419 0.219185
```

The full list of program options are:
```
-h          Display help message
-v          Verbose mode, output more updates to STDOUT
-g path*    Path to the input graph on the Matrix Market (mtx) format
-q          Alternate mode to quickly compute colors used by each heuristic
-f          Only run FF, LF, SL, and GNN (no ID, SD, or LOG versions)
-c #        Number of repeated colorings using Culberson
-t #        Number of threads to use for faster setup
```

\* Mandatory input

The Heuristics are named as follows:
- FF - First Fit
- LF - Largest Degree First
- LLF - Largest Log Degree First
- SL - Smallest Degree Last
- SLL - Smallest Log Degree Last
- ID - Incidence Degree
- SD - Saturation Degree
- SDA - Alternate Saturation Degree
- GNN2 - GNN 2 layers
- GNN3 - GNN 3 layers
- GNN4 - GNN 4 layers
