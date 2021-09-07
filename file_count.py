import csv

# with open('trainA.csv') as f:
#     reader = csv.reader(f)
#     i = 0
#     for row in reader:
#         i += 1
#     print(i)

with open('testA.csv') as f:
    reader = csv.reader(f)
    i = 0
    for row in reader:
        i += 1
    print(i)