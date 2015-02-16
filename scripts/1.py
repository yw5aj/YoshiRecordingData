import os
os.chdir('x:/WorkFolder/AbaqusFolder/YoshiModel/')


import numpy as np
from abqimport import *
from setfiber import Fiber, getStimBlockFromCsv
from feconstants import materialBlockFiber as materialBlock


stimBlock = getStimBlockFromCsv('x:/WorkFolder/DataAnalysis/YoshiRecordingData/csvs/stim_block_1.csv')
fiber = Fiber(baseModelName='Fiber1', suffix='', stimBlock=stimBlock, materialBlock=materialBlock, runFiber=True, doAnalysis=False, skipWait=True)
np.savetxt('./csvs/'+fiber.baseModelName+'StaticForceDispl.csv', np.column_stack((fiber.staticDisplList, fiber.staticForceList)), delimiter=',')
for i, model in enumerate(fiber.modelList):
    output = np.c_[model.time, model.force, model.displ, model.stress, model.strain, model.sener]
    np.savetxt('./csvs/'+fiber.baseModelName+'Output'+str(i)+'.csv', output, delimiter=',')


import sys
sys.exit()

