#!/bin/bash

docker run -v /Users/Douglas.Hindson/workspace/story_visualizer:/deepdream/deepdream/files saturnism/deepdream:latest python /deepdream/deepdream/files/story_visualizer/deepdream.py "$@"