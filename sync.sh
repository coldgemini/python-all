#!/usr/bin/env bash
rsync -avuhP  -e "ssh -p 10822" ~/Dev/python-all/* zhouxiangyong@s108:~/Dev/python-all/
