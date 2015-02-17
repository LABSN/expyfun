.. -*- mode: rst -*-

.. role:: bash(code)
   :language: bash

Recommended GitHub workflow
===========================
Typically there are three types of people who use GitHub to access source code:
users, developers, and maintainers. Users_ simply want access to the latest
code. Developers_ want to make some changes to the code. Maintainers_ are the
ones who make sure the changes to the code make sense, and incorporate them
into the code base. We will talk about each of these use cases in turn.

One of the most challenging concepts with :bash:`git` source code control
is the fact that its job is to seamlessly manage multiple different versions
of a code base. The idea is that :bash:`git` allows you to swap back and
forth between these different versions while working within the same directory
structure.

In other words, typically there is only one repository for a given project on
a user's local machine. That repository can have multiple branches within it.
When a user switches between branches, they are effectively switching between
different sets of files within that repository. The :bash:`git` protocol thus
modifies or updates files in-place for users.

In addition to each user having a local git repository, they usually become
associated with some number of remote repositories on GitHub. The user's local
repository can "know" about any number of remote repositories, and users get 
to call those remote repositories whatever they want. Here is a diagram of the
connections between the remote repositories on GitHub and the local computers 
of each of three different users:

.. _diagram:
.. image:: https://cdn.rawgit.com/LABSN/expyfun/master/doc/git_flow.svg


Users
^^^^^
Users who want access to the source code typically do something like this::

    $ git clone git://github.com/LABSN/expyfun.git
    $ cd expyfun
    $ python setup.py install

Perhaps without realizing it, these users have set up a structure so that
their local repository knows about a remote repository named "origin" that
points to `<http://github.com/LABSN/expyfun.git>`_, and they have checked out
the branch "master" from that repository. This can be seen by doing::

    $ git remote -v
    origin	git://github.com/LABSN/expyfun.git (fetch)
    origin	git://github.com/LABSN/expyfun.git (push)
    $ git branch
    * master

This setup is convenient, because to update their local repo they can just do::

    $ git pull

Under the hood, this is doing something more like this::

    $ git pull origin/master

And thus they can keep code up to date with the remote repo easily.


Developers
^^^^^^^^^^
Developers typically use a slightly different setup. The first step is to go
onto GitHub.com and fork the repository so that changes can be sandboxed.
If the developer had account :bash:`rkmaddox` It is more standard to
have :bash:`origin` point to their own fork. The easiest way to do this is::

    $ git clone git@github.com:/rkmaddox/expyfun.git
    $ cd expyfun
    $ git remote add upstream git://github.com/LABSN/expyfun.git

Now they are set up with the standard :bash:`origin`/:bash:`upstream` flow
as::

    $ git remote -v
    origin	git@github.com:/rkmaddox/expyfun.git (fetch)
    origin	git@github.com:/rkmaddox/expyfun.git (push)
    upstream	git://github.com/LABSN/expyfun.git (fetch)
    upstream	git://github.com/LABSN/expyfun.git (push)
    $ git branch
    * master

Now to make a change, the flow would be something like this (to e.g., add
some trivial file)::

    $ git branch fix_branch
    $ git branch
      fix_branch
    * master
    $ git checkout fix_branch
    $ git branch
    * fix_branch
      master
    $ touch foo.txt
    $ git add foo.txt
    $ git commit -am 'FIX: Add missing foo file'
    $ git push origin fix_branch

This creates a new branch called :bash:`fix_branch` on the local machine,
checks out that branch, adds a file, commits the change to the branch, and
then pushes the branch to the :bash:`origin` repo. The user could then
navigate to `<http://github.com/LABSN/expyfun/>`_ and they would find a nice
**Pull Request** button available to open a pull request.

Maintainers_ would then typically comment on the pull request and ask for
some changes. For example, maybe the user forgot to also add the necessary
:bash:`bar` file. The user would then do::

    $ git branch
    * fix_branch
      master
    $ touch bar.txt
    $ git add bar.txt
    $ git commit -am 'FIX: Add missing file'
    $ git push origin fix_branch

After this set of commands, the pull request (PR) is automatically 
updated to reflect this new addition. The cycle of commenting on and 
updating the continues until the Maintainers_ are satisfied with the 
changes. They will then merge the pull request to incorporate the 
proposed changes into the GitHub repo.

Once their branch gets merged into the :bash:`master` branch of
`<github.com/LABSN/expyfun>`, the developer can do the following to get
up to date on their local machine::

    $ git checkout master
    $ git fetch upstream
    $ git pull upstream/master
    $ git branch -d fix_branch
    $ git branch
    * master


Maintainers
^^^^^^^^^^^
Maintainers start out with a similar set up as users. However, they might
want to be able to push directly to the :bash:`upstream` repo. Having a 
repo set up with :bash:`git://` access instead of :bash:`git@github.com`
or :bash:`https://` access will not allow pushing. So starting from
scratch, a maintainer might do::

    $ git clone git@github.com:/Eric89GXL/expyfun.git
    $ cd expyfun
    $ git remote add upstream git@github.com:/LABSN/expyfun.git
    $ git remote add ross git://github.com/rkmaddox/expyfun.git

Now the maintainer's local repository knows about their own personal
development fork, the upstream repo, and :bash:`rkmaddox`'s fork::

    $ git remote -v
    origin	git@github.com:/Eric89GXL/expyfun.git (fetch)
    origin	git@github.com:/Eric89GXL/expyfun.git (push)
    ross	git://github.com/rkmaddox/expyfun.git (fetch)
    ross	git://github.com/rkmaddox/expyfun.git (push)
    upstream	git://github.com/LABSN/expyfun.git (fetch)
    upstream	git://github.com/LABSN/expyfun.git (push)

Let's say :bash:`rkmaddox` has opened a PR on Github, and the maintainer wants
to test out the code. This can be done this way::

    $ git fetch ross
    $ git checkout -b ross_branch ross/fix_branch

The first command allows the local repository to know about the changes (if
any) that have occurred out on Github at `<github.com/rkmaddox/expyfun.git>`_.
In this case, a new branch named :bash:`fix_branch` has been added.

The second command is more complex. :bash:`git checkout -b $NAME` is a command
that first creates a branch named :bash:`$NAME`, then checks it out. The 
additional argument :bash:`ross/fix_branch` tells :bash:`git` to make the
branch track changes from the remote branch :bash:`fix_branch` in the remote
repository known as :bash:`ross`, which you may recall points to
`<github.com/rkmaddox/expyfun.git>`_. The full command can thus be interpreted
in human-readable form as "create and check out a branch named
:bash:`ross_branch` that tracks the changes in the branch
:bash:`fix_branch` from the remote repo named :bash:`ross`".

Once the code is merged on GitHub, the maintainer can update their local copy
in a similar way as the developer did earlier::

    $ git checkout master
    $ git pull upstream/master
