# dropbox_syncer
Synchronize a local directory with a Dropbox folder using the Dropbox Python SDK.

By default, this script will act in a non-destructive way, and only add new files to the local and remote folders.

It will never delete a file.

Optionally, if the same file is present on both sides, it will compare their modification times and keep the most recent (command-line option `-O`).

### Dropbox App Creation

First, you need to create a Scoped App from https://www.dropbox.com/developers/, using permissions:
`files.content.write` and `files.content.read`. If you just want to download from Dropbox (option -D),
only `files.content.read` is necessary.

If you need access to only one folder, you can create a scoped app with an App Folder.

### Installation

The script can either be downloaded and executed directly, or installed using pip: `pip install https://github.com/devernay/dropbox_syncer.git`

`pip` will install a script named `dropbox-syncer`.

### Usage

```
usage: dropbox_syncer.py [-h] [--remote-folder REMOTE_FOLDER] [--local-folder LOCAL_FOLDER]
                         [--auth-file AUTH_FILE] [--app-key APP_KEY] [--app-secret APP_SECRET] [--dry-run]
                         [-D] [-U] [-O]

Dropbox Two-Way Sync Tool

options:
  -h, --help            show this help message and exit
  --remote-folder REMOTE_FOLDER
                        Dropbox folder path
  --local-folder LOCAL_FOLDER
                        Local folder path
  --auth-file AUTH_FILE
                        Authentication file path
  --app-key APP_KEY     Dropbox app key (optional)
  --app-secret APP_SECRET
                        Dropbox app secret (optional)
  --dry-run             Do not do any actual action, only write the log file of what would be done.
  -D, --download        Download from Dropbox.
  -U, --upload          Upload to Dropbox.
  -O, --overwrite       If file exists at both places but differ, overwrite either version based on their
                        modification time.
```

The first time you run the script, you have to pass the App key and the App secret on the command-line.
The script will ask you to authenticate using a browser, and then store credentials in the auth file.
Subsequent runs will use the auth file.  Make sure that the auth file is stored in a safe place, as
it gives access to your Dropbox.