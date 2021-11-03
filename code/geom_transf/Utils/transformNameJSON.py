import os

src = "C:\\Users\\Eros Bignardi\\projectCV\\data\\test\\pose"


def rename_point(src):
    for file in os.listdir(src):
        name = ""
        if os.path.isdir(os.path.join(src, file)):
            rename_point(os.path.join(src, file))
        if "." not in file:
            name = file[:-4] + ".json"
            os.rename(os.path.join(src, file), os.path.join(src, name))
            print(name)


if __name__ == '__main__':
    rename_point(src)
