#!/bin/bash

echo '------- update git and remote --------'

git add .

git commit . -m 'paper accepted by tnnls'

git push origin master

echo '------- update complete --------'