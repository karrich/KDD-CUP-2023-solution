# Co-occurrence matrix
When implementing it, we built a variety of different co-occurrence matrices.
# Single-hop recall
Suppose P(i,·) is the co-occurrence set of item `i` to other items in the cooccurrence matrix, then the recall formula is:

![image](https://github.com/karrich/KDD-CUP-2023-solution/assets/57396778/770bb44b-1821-489a-b342-5b185bd2bf7c)

`W` is the weight, which is determined by the position of the item in the session.
# Multi-hop recall
The recall formula for multi-hop recall and single-hop recall is the same,
except that the matrix of multi-hop recall is generated from the matrix of single-hop recall:

![image](https://github.com/karrich/KDD-CUP-2023-solution/assets/57396778/e2521824-344b-4ef2-9cd1-8eb5feee8431)

# detail
We found that for a session, we want to predict the label, it must not have appeared in the session.
