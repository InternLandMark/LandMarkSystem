#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
from datetime import datetime


def searchForMaxIteration(folder):
    saved_iters = [int(fname.split("_")[-1]) for fname in os.listdir(folder)]
    return max(saved_iters)


def set_print_with_timestamp(print_timestamp):
    old_f = sys.stdout

    class F:
        """Class F"""

        def __init__(self, print_timestamp):
            self.print_timestamp = print_timestamp

        def write(self, x):
            if self.print_timestamp:
                if x.endswith("\n"):
                    timestamp = datetime.now().strftime("%d/%m %H:%M:%S")
                    old_f.write(x.replace("\n", f" [{str(timestamp)}]\n"))
                else:
                    old_f.write(x)

            else:
                old_f.write(x)

        def flush(self):
            old_f.flush()

    sys.stdout = F(print_timestamp)
