import sys
import os

if __name__ == "__main__":
    path1 = sys.argv[1]
    path2 = sys.argv[2]

    reviews1 = []
    with open(path1, "r", encoding='utf-8') as f:
        reviews1 = [line for line in f]
    
    reviews2 = []
    with open(path2, "r", encoding='utf-8') as f:
        reviews2 = [line for line in f]

    for i in range(len(reviews1)):
        if len(reviews1[i]) != len(reviews2[i]):
            print(i, reviews1[i], reviews2[i])
            break
