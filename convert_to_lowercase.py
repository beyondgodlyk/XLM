#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os

if __name__ == "__main__":
    path = sys.argv[1]
    assert os.path.isfile(path)
    with open(path, "r", encoding='utf-8') as f:
        for line in f:
            print(line.lower(), end="")
    
