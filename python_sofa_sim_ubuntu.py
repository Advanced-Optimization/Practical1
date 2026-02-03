"""Testing stuff."""

import argparse
import os
import sys
sys.path.append(os.path.join(os.environ["SOFA_ROOT"], "plugins", "SofaPython3", "lib", "python3", "site-packages"))

import numpy as np
import Sofa
import Sofa.SofaGL

from modules.pytorch_mlp import PytorchMLPReg
from sofa_sim import createScene

from OpenGL.GL import *
from OpenGL.GLU import *
import pygame


resultsDirectory = os.path.dirname(os.path.realpath(__file__))

class RenderingController(Sofa.Core.Controller):

    def __init__(self, rootnode):
        Sofa.Core.Controller.__init__(self)
        self.name = "RenderingController"
        self.rootnode = rootnode

        self.width = 1600
        self.height = 1000
        self.screen_size = (self.width, self.height)

        pygame.init()
        pygame.display.set_mode((self.width, self.height))
        pygame.display.init()
        pygame.font.init()

        self.screen = pygame.display.set_mode(self.screen_size, pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE)
        self.surface = pygame.Surface((self.width, self.height))
    
    def __del__(self):
        pygame.quit()
    
    def onAnimateBeginEvent(self, _): 
        pygame.event.get()

    def onAnimateEndEvent(self, _):
        self.showSimulation()

    def showSimulation(self):

        glClearColor(0.76, 0.78, 0.80, 1.0)
        glViewport(0, 0, self.width, self.height)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glEnable(GL_LIGHTING)
        glEnable(GL_DEPTH_TEST)

        Sofa.SofaGL.glewInit()
        Sofa.Simulation.initVisual(self.rootnode)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, (self.width / self.height), 15, 2000)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        cameraMVM = self.rootnode.camera.getOpenGLModelViewMatrix()

        glMultMatrixd(cameraMVM)
        Sofa.SofaGL.draw(self.rootnode)

        buff = glReadPixels(0, 0, self.width, self.height, GL_RGB, GL_UNSIGNED_BYTE)

        image_array = np.fromstring(buff, np.uint8)
        if image_array.shape != (0,):
            image = image_array.reshape(self.height, self.width, 3)
        else:
            image = np.zeros((self.height, self.width, 3))

        image = np.flipud(image)
        image = np.moveaxis(image, 0, 1)

        # Update the window
        self.surface = pygame.surfarray.make_surface(image)
        self.screen.blit(self.surface, (0, 0))

        # Display the modifications
        pygame.display.flip()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog=sys.argv[0], description="Simulate a leg.")
    parser.add_argument(
        metavar="model_file",
        type=str,
        help="the path to the file containing the model",
        dest="model_file",
    )
    args = parser.parse_args()

    print("model file:", args.model_file)
    args.model_file = resultsDirectory + "/" + args.model_file
    sys.argv[1] = args.model_file

    rootnode = Sofa.Core.Node("rootnode")
    createScene(rootnode=rootnode)
    cameraPosition = [80.71, 3.92, 2.32]
    rootnode.addObject("LightManager")
    rootnode.addObject("DirectionalLight", direction=[1, 0, 1])
    rootnode.addObject("InteractiveCamera", name='camera', 
                       position=cameraPosition, 
                       orientation=[-0.372, 0.574, 0.309, 0.661],
                       lookAt=[-25.37, -166, -11])
    rootnode.addObject("VisualStyle", displayFlags="showVisualModels")
    rootnode.addObject(RenderingController(rootnode=rootnode))

    Sofa.Simulation.init(rootnode)
    regr = rootnode.MLPController.regr

    dt = 0.01
    for step in range(2500):
        Sofa.Simulation.animate(rootnode, dt)
