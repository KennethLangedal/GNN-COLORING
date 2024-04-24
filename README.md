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
./coloring [options] tc1 tc2 ...
```

For example:
```
./coloring -g [path to graph] -t 16 -p -v 1 16
```
This will run parallel experiments using 1 and 16 threads.

The full list of program options are:
* -g path, sets input graph file
* -p, run parallel experiments
* -c, quickly compute colors used by each heuristic
* -k #, number of repeated colorings
* -t #, total number of threads for faster I/O and validation
* -v, verbose
* -h, print help message

