# -*- coding: utf-8 -*-
"""XML writing functions
"""

from os import path as op


HEADER = """\
<?xml version="1.0"?>
<AUDIOFILE_PLAYLIST>
   <PLAYLIST1>
      <Entry Folder="%s"/>
"""
ENTRY = """\
      <Entry Code%03d="%s"/>
"""
FOOTER = """\
   </PLAYLIST1>
   <SYSTEM>
      <Entry UseDigitalInputD0="FALSE"/>
      <Entry StopCode="19"/>
      <Entry SwapChannels="FALSE"/>
      <Entry SDRAMTest="FALSE"/>
   </SYSTEM>
</AUDIOFILE_PLAYLIST>
"""


def write_playlist(fnames, playlist_fname='Playlist.xml',
                   subdirectory='Audio Tracks', overwrite=False):
    """Write an XML playlist for use with an AudioFile.

    Parameters
    ----------
    fnames : list of str
        The filenames.
    playlist_fname : str | None
        The output playlist filename.
        Must end with 'Playlist.xml'.
    overwrite : bool
        If True, overwrite the existing file.
    """
    if not op.basename(playlist_fname) == 'Playlist.xml':
        raise ValueError('playlist_fname must end with "Playlist.xml", got %r'
                         % (playlist_fname,))
    if not overwrite and op.isfile(playlist_fname):
        raise IOError('File {} exists, overwrite=True must be '
                      'used'.format(op.basename(playlist_fname)))
    if not isinstance(fnames, (list, tuple)):
        raise TypeError('fnames must be a list of str, got type %s'
                        % (type(fnames),))
    if not isinstance(subdirectory, str) or '\\' in subdirectory or \
            '/' in subdirectory:
        raise ValueError('subdirectory must be str with no slashes, got type '
                         '%s (%r)' % (type(subdirectory), subdirectory))
    for fi, fname in enumerate(fnames):
        if not isinstance(fname, str):
            raise TypeError('Entries in fnames must be str, got fnames[%d]'
                            ' as type %s (%r)' % (fi, type(fname), fname))
        if '/' in fname or '\\' in fname:
            raise ValueError('Entries in fnames cannot have slashes, got '
                             'fnames[%d] = %r' % (fi, fname))
        if not op.splitext(fname)[1].lower() == '.wav':
            raise ValueError('Entries in fnames must end with .wav, got '
                             'fnames[%d] = %r' % (fi, fname))
    with open(playlist_fname, 'w') as fid:
        fid.write(HEADER % subdirectory)
        for fi, fname in enumerate(fnames):
            fid.write(ENTRY % (fi + 1, fname))
        fid.write(FOOTER)
