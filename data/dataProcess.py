import os

path = r"./list_68pt_rect_attr_train_test/list_68pt_rect_attr_train.txt"

with open(path, "r") as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip().split(" ")
        print(line)
        print(len(line))
        # print(line.split(" "))
        break