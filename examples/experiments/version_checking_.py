"""
==========================
Version checking functions
==========================

This demonstrates how the version checking functions work.
"""
# Author: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD (3-clause)

import tempfile

from expyfun import download_version, run_subprocess

print(__doc__)


# Let's say we want to fix our experiment to use a specific version of
# expyfun. First we'd want to install that version (referenced by the
# commit number) to the proper directory. Here we'll use a temporary
# directory so we don't break any other code examples, but usually you'd
# want to do it in the experiment directory:
temp_dir = tempfile.mkdtemp()
download_version("c18133c", temp_dir)

# Now we would normally need to restart Python so the next ``import expyfun``
# call imported the proper version. We'd want to add an ``assert_version``
# call to the top of our script We can simulate that here just by
# launching a new Python instance in the ``temp_dir`` and using our assertion
# function:

cmd = """
from expyfun import assert_version

assert_version('c18133c')
"""
try:
    run_subprocess(["python", "-c", cmd], cwd=temp_dir)
except Exception as exp:
    print(f"Failure: {exp}")
else:
    print("Success!")

# Try modifying the commit number to something invalid, and you should
# see a failure.
