from contextlib import contextmanager
import inspect
import itertools
import json
import os
import re
import runpy
import sys


class ModelConfigToUpdate:
    def __init__(self, path, new_path=None):
        self.path = path
        self.new_path = new_path
    
    def __enter__(self):
        self.config_dict = json.loads(Path(self.path).read_text())
        return self.config_dict

    def __exit__(self, *args):
        new_path = self.new_path or self.path
        Path(new_path).write_text(json.dumps(self.config_dict))
        
        
def create_variables_in_module(dictionary):
    caller_module = get_caller_module()
    vars(caller_module).update(dictionary)
    
    
def get_caller_module():
    caller_frame = inspect.stack()[2]
    caller_module_name = caller_frame.frame.f_globals['__name__']
    return sys.modules[caller_module_name]
    
    
def get_module_declared_variables():
    def is_imported(variable_name):
        return any(
            re.search(fr"(?:import\s+{variable_name}\b)|(?:from\s.+\simport\s.+\sas\s{variable_name}\b)", line)
            for line in source_lines
        )

    def is_created(variable_name):
        return any(
            re.match(fr"{variable_name}\s*=", line)
            for line in source_lines
        )

    def is_declared_in_caller_module(name):
        return not is_imported(name) and is_created(name)
        
    caller_module = get_caller_module()
    variables = vars(caller_module)
    source_lines = inspect.getsourcelines(caller_module)[0]
        
    return {
        name: value
        for name, value in vars(caller_module).items()
        if is_declared_in_caller_module(name)
    }


class WorkingDirectoryOn:
    def __init__(self, working_directory):
        self.working_directory = working_directory
        self.original_working_directory = os.getcwd()
        
    def __enter__(self):
        os.chdir(self.working_directory)
    
    def __exit__(self, exception_type, exception_instance, traceback):
        os.chdir(self.original_working_directory)


@contextmanager
def cli_arguments(**arguments):
    original_arguments = sys.argv
    sys.argv = kwargs_to_command_line_arguments(**arguments)
    try:
        yield
    finally:
        sys.argv = original_arguments


def kwargs_to_command_line_arguments(**kwargs):
    return [None] + list(itertools.chain.from_iterable([
        (f"--{key}", str(value))
        for key, value in kwargs.items()
    ]))


def run_module_as_main(module, from_directory, command_line_arguments):
    with (
        WorkingDirectoryOn(from_directory),
        cli_arguments(**command_line_arguments)
    ):
        return runpy.run_module(module, run_name='__main__', alter_sys=True)