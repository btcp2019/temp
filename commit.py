import os
import sys

s = 'update'

os.system('git pull')
os.system('git add -A')
os.system('git commit -m \'' + s + '\'')
os.system('git push')
