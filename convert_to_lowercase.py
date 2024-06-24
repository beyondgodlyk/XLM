import sys

if __name__ == "__main__":
    with open(sys.argv[1], "r", encoding='utf-8') as f:
        for line in f:
            print(line.lower(), end="")
    
