import os
import shutil
from random import sample
rootPath=os.path.join(os.getcwd(),"data_sub")
dirs=os.listdir(rootPath)
sub30=sample(dirs,3)

subDir=os.path.join(os.getcwd(),"data_sub_30")
if not os.path.exists(subDir):
    os.makedirs(subDir)
shutil.rmtree(subDir)
for i in range(0,len(sub30)):
    now_path=os.path.join(rootPath,sub30[i])
    shutil.copytree(now_path,os.path.join(subDir,sub30[i]))