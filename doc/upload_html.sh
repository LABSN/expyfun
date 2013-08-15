#!/usr/bin/env bash

REMOTE_PATH="/lvmraid/www/html/expyfun/"
rsync -rltvz --delete --perms --rsh="ssh -p 2222" --chmod=g+w build/html/ lester.ilabs.uw.edu:$REMOTE_PATH
ssh -p 2222 lester.ilabs.uw.edu "chgrp -R apache $REMOTE_PATH"
