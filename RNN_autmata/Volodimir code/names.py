path = "./names.txt"

def lang_names(path):
    names = []
    extensions = []
    with open(path) as file:
        length = len(file.readlines())
    with open(path) as file:
        A = True
        h = 0
        while A:
            line = file.readline()
            h+=1
            A = not(line=='\n')
            if A:
                names.append(line[:-1])
        for j in range(length-h):
            line = file.readline()
            extensions.append(line[:-1])
    return (names,extensions)

names, ext = lang_names(path)
print(names,ext)