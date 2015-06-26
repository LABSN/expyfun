Git command Cheat Sheet! Become one with the Git…

Quick terms::
	Local:
	Remote:

Creating repositories::
	existing repository (clone):
		
		$ git clone ssh://user@domain.com/repo.git

		This copies a remote repository to create a local repository with a remote 			called 'origin'

	new local repository:

		$ git init

Local changes::

	$ git status
	
	Check the status of changed files in your current working directory

	$ git add
	
	Tells git to keep track of all changes to this file

	$ git commit

	This takes what you have told git to save by using 'git add' and stores a copy 			permanently inside .git dr. This copy is a 'revision'.

	$ git diff

	Shows changes between current state and the most recent saved version.

Updating (push and pull)::

	$ git remote -v

	Gives a list of current remotes and names (origin, master, ect.)

	$ git remote add <shortname> <url>

	This will add a new repository, named <remote>

	$ git fetch <remote>

	Downloads changes from <remote>, does not integrate into HEAD

	$ git pull <remote> <branch>

	Copies from remote to local repository(merges) into HEAD

	$ git push <remote> <branch>

	This copies changes from local to remote repository

Navigating Branches::

	$ git branch -av

	Gives a list of all existing branches

	$ git checkout <branch>

	Switches HEAD branch

	$ git branch <new-branch>

	Creates a new branch based on your currend HEAD


Checking history::

	$ git log 

	Shows history of changes in reverse order

Undo::

	$ git checkout HEAD <file>

	Discards local changes in <file>

	$ git revert <commit>

	Reverses a commit by producing a new commit with contrary changes.
