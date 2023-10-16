import git
from IPython.core.magic import register_cell_magic, register_line_magic
from pathlib import Path
import os

from jinja2 import Environment, select_autoescape


def load_ipython_extension(ipython):
    @register_cell_magic
    def rendertemplate(output, template_text):
        notebook_variables = ipython.user_ns
        
        env = Environment(
            variable_start_string='[[',
            variable_end_string=']]',
        )
        template = env.from_string(template_text)
        rendered_text = template.render(**notebook_variables)

        Path(output).write_text(rendered_text)
        return output

    @register_line_magic
    def cd_repo_root(line):
        repository_root = git.Repo(".", search_parent_directories=True).working_tree_dir
        os.chdir(repository_root)
        return repository_root
