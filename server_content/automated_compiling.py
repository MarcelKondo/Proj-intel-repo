
import os
import subprocess

def define_copiler_settings(opLevel, simdType,n1 = 256 ,n2 = 256 ,n3 = 256 , nb_threads = 4, nb_it = 100 , 
                            tblock1 = 32 ,tblock2 = 32, tblock3 = 32,):
    #os.system("cd ~/Proj-Intel/Appli-iso3dfd/")
    #os.system("pwd")
    os.system("cd ~/Proj-Intel/Appli-iso3dfd/ && make -e OPTIMIZATION=\"-O"+ str(opLevel) + "\" -e simd=" + simdType + " last" )
    #make -e OPTIMIZATION="-O3" -e simd=sse last

    os.system("cd ~/Proj-Intel/Appli-iso3dfd/bin")
    os.system("pwd")
    res = subprocess.run("cd ~/Proj-Intel/Appli-iso3dfd/bin && iso3dfd_dev05_cpu_" + simdType + ".exe", shell=True,
                         stdout=subprocess.PIPE)
    print(str(res.stdout,'utf-8'))
    print(type(res.stdout))
    print("testeeeeeee\naaaaaaaaaaa")
    print(type("aa"))
    return res.stdout
print("V2 starting execution")
define_copiler_settings(opLevel = 3, simdType = "sse")
print("execution finished")
