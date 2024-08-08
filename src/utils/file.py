#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import hashlib
import pathlib


def check_dir(dir_path):
    """Create a directory if it doesn't exist"""

    if isinstance(dir_path, pathlib.PosixPath):
        if not dir_path.exists():
            dir_path.mkdir(parents=True)
    elif isinstance(dir_path, pathlib.WindowsPath):
        if not dir_path.exists():
            dir_path.mkdir(parents=True)
    elif isinstance(dir_path, str):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    else:
        assert False


def list_dirs(dir_path):
    """List all directories"""

    dir_paths = []
    for root, dirs, files in os.walk(dir_path):
        for name in dirs:
            dir_paths.append(os.path.join(root, name))
    dir_paths.sort()
    return dir_paths


def list_files(dir_path):
    """List all files"""

    file_paths = []
    for root, dirs, files in os.walk(dir_path):
        for name in files:
            file_paths.append(os.path.join(root, name))
    file_paths.sort()
    return file_paths


def get_exp_dir(root_dir_path, *args):
    """Get the result directory"""

    check_dir(root_dir_path)
    dir_name = ""
    for i, arg in enumerate(args):
        if i == 0:
            dir_name += str(arg)
        elif arg:
            dir_name += f"_{str(arg)}"
    exp_dir_path = os.path.join(root_dir_path, dir_name)
    return exp_dir_path


def compute_md5(raw_bytes):
    """Compute the md5 hash"""

    md5_obj = hashlib.md5()
    md5_obj.update(raw_bytes)
    return md5_obj.hexdigest()


