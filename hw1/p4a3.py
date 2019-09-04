#!/usr/bin/env python3

import p4a2

if __name__ == "__main__":
    p4a2.lasso_solve("baboon.csv", [10**(-k/4) for k in range(9)])
