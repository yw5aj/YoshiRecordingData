import os
os.chdir('x:/YuxiangWang/AbaqusFolder/YoshiModel/')


import numpy as np
from abqimport import *
from setfiber import Fiber, getStimBlockFromCsv, materialBlockDefault


materialBlock = materialBlockDefault
stimBlock = getStimBlockFromCsv('x:/YuxiangWang/DataAnalysis/YoshiRecordingData/csvs/stim_block_4.csv')
fiber = Fiber(baseModelName='Fiber4', suffix='', stimBlock=stimBlock, materialBlock=materialBlock, runFiber=True, doAnalysis=False, skipWait=True)
np.savetxt('./csvs/'+fiber.baseModelName+'StaticForceDispl.csv', np.column_stack((fiber.staticDisplList, fiber.staticForceList)), delimiter=',')
for i, model in enumerate(fiber.modelList):
    output = np.c_[model.time, model.force, model.displ, model.stress, model.strain, model.sener]
    np.savetxt('./csvs/'+fiber.baseModelName+'Output'+str(i)+'.csv', output, delimiter=',')


import sys
sys.exit()

