#!/bin/bash

echo '------- update git and remote --------'

git add .

git commit . -m 'add a toy example'

git push origin master

echo '------- update complete --------'