#/usr/bin/env python

from __future__ import division

import sys

from . import MPI
comm = MPI.comm

class StatusBar(object):
    def __init__(self, count, width=60):
        self.counter = 0
        self.count = count
        self.width = width

        self.format = '\r[%%-%ds] %%3d%%%%' % self.width

        self.show()

    def update(self):
        self.counter += 1

        self.show()

    def show(self):
        if comm.rank:
            return

        progress = self.counter / self.count

        sys.stdout.write(self.format % ('=' *
            int(round(progress * self.width)),
            int(round(progress * 100))))

        if self.counter == self.count:
            sys.stdout.write('\n')
