#!/usr/bin/env python
# -*- coding: utf-8 -*-


def print_progress(step, max_step):
    """
    Subroutine prints percentage of progress
    :param step: Current step value
    :param max_step: Maximum value
    :return: None
    """
    import sys
    sys.stdout.write("\r%d%%" % (100 * float(step) / float(max_step)))
    sys.stdout.flush()


def progress_bar(step, max_step):
    """
    Subroutine prints simple progress bar with percentage
    :param step: Current step value
    :param max_step: Maximum value
    :return: None
    """
    progress = int(100 * float(step) / float(max_step))
    print '\r[{0}] {1}%'.format('#'*(progress/10), progress),
