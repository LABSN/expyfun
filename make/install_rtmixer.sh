#!/bin/bash

pushd ~
git clone https://github.com/spatialaudio/python-rtmixer --depth=50
cd python-rtmixer
git submodule init
git submodule update
pip install -v "$@" .
popd
