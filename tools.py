import os

def compile_program(args=''):
    os.system("(cd ../Appli-iso3dfd/ && make " + args + " last)")