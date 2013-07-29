# -*- coding: utf-8 -*-
"""
Script to generate an experiment configuration file.
Created on Mon Jul 29 11:28:06 2013
@author: Daniel McCloy (drmccloy@uw.edu)
"""

import logging

from .utils import set_config, verbose

logger = logging.getLogger('expyfun')


def _check_TDT():
    is_TDT = raw_input('Enter 1 for TDT or 0 for internal soundcard: ')
    if is_TDT == '1':
        return True
    elif is_TDT == '0':
        return False
    else:
        return None


def _check_type():
    tdtType = raw_input('Which TDT model number (e.g., RM1, RP2, etc.)? ')
    if tdtType in ['RA16', 'RL2', 'RM1', 'RM2', 'RP2', 'RV8', 'RX5', 'RX6',
                   'RX7', 'RX8', 'RX9', 'RZ2', 'RZ3', 'RZ4', 'RZ5', 'RZ6']:
        return tdtType
    else:
        return False


def _check_interface():
    tdtInterface = raw_input('How is TDT connected? Enter 1 for USB or '
                             '0 for Gigabit: ')
    if tdtInterface == '1':
        return 'USB'
    elif tdtInterface == '0':
        return 'GB'
    else:
        return None


def _check_resp_dev():
    responseDevice = raw_input('Enter 1 for keyboard or 0 for buttonbox: ')
    if responseDevice == '1':
        return 'keyboard'
    elif responseDevice == '0':
        return 'buttonbox'
    else:
        return None


@verbose
def create_system_config(verbose=None):
    """Initialize system settings

    This function will initialize the computer setup script by prompting
    the user to enter information, and storing it in a permanent file.
    """
    logger.info('System profiler commencing: ')

    # check for TDT
    is_TDT = _check_TDT()
    while is_TDT is None:
        is_TDT = _check_TDT()

    # get TDT type
    if is_TDT:
        # validate type
        tdt_type = _check_type()
        while not _check_type():
            tdt_type = _check_type()

        # get interface
        tdt_interface = _check_interface()
        while _check_interface() is None:
            tdt_interface = _check_interface()
        # get circuit path
        tdt_circuit = raw_input('Type or paste in the path to your TDT '
                                'circuit: ')
    else:
        tdt_type = 'psychopy'
        tdt_interface = None
        tdt_circuit = None

    # get response device type
    response_device = _check_resp_dev()
    while response_device is None:
        response_device = _check_resp_dev()

    # collect all into dict
    set_config('AUDIO_CONTROLLER', tdt_type)
    set_config('TDT_INTERFACE', tdt_interface)
    set_config('TDT_CIRCUIT', tdt_circuit)
    set_config('RESPONSE_DEVICE', response_device)
