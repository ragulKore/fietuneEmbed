import os
def get_file_paths():
    paths=[]
    for r, d, f in os.walk(r'./data/train'):
        for file in f:
            if '.pdf' in file:
                paths.append(os.path.join(r, file))
    print(paths[0])
def training():
    print('hello')

if __name__ == "__main__":
    get_file_paths()