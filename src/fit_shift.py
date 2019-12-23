#!/usr/bin/env python

import sys
import template


def main():
    lstname = sys.argv[1]
    namelst = [i.strip() for i in open(lstname)]
