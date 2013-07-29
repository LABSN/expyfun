# -*- coding: utf-8 -*-
"""
Script to generate an experiment configuration file.
Created on Mon Jul 29 11:28:06 2013
@author: Daniel McCloy (drmccloy@uw.edu)
"""

from .utils import set_config

def checkTDT():
    isTDT = raw_input('Enter 1 for TDT or 0 for internal soundcard: ')
    if isTDT is 1: return(True)
    elif isTDT is 0: return(False)
    else: return(None)

def checkType():
    tdtType = raw_input('Which TDT model number (e.g., RM1, RP2, etc.)? ')
    if tdtType in ['RA16','RL2','RM1','RM2','RP2','RV8','RX5','RX6','RX7',
                   'RX8','RX9','RZ2','RZ3','RZ4','RZ5','RZ6']:
        return tdtType
    else: return False

def checkInterface():
    tdtInterface = raw_input('How is TDT connected? Enter 1 for USB or 0 for Gigabit: ')
    if tdtInterface is 1: return('USB')
    elif tdtInterface is  0: return('GB')
    else: return(None)

def checkRespDev():
    responseDevice = raw_input('Enter 1 for keyboard or 0 for buttonbox: ')
    if responseDevice is 1: return('keyboard')
    elif responseDevice is 0: return('buttonbox')
    else: return(None)

print('System profiler commencing: ')
# check for TDT
isTDT = checkTDT()
while isTDT is None:
    isTDT = checkTDT()

# get TDT type
if isTDT:
    # validate type
    tdtType = checkType()
    while not checkType():
        tdtType = checkType()
    # get interface
    tdtInterface = checkInterface()
    while checkInterface() is None:
        tdtInterface = checkInterface()
    # get circuit path
    tdtCircuit = raw_input('Type or paste in the path to your TDT circuit: ')

else:
    tdtType = 'psychopy'
    tdtInterface = None
    tdtCircuit = None

# get response device type
responseDevice = checkRespDev()
while responseDevice is None:
    responseDevice = checkRespDev()

# collect all into dict
set_config('AUDIO_CONTROLLER',tdtType)
set_config('TDT_INTERFACE',tdtInterface)
set_config('TDT_CIRCUIT',tdtCircuit)
set_config('RESPONSE_DEVICE',responseDevice)

