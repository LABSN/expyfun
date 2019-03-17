import os.path as op
import xml.etree.ElementTree as ET

import pytest

from expyfun.io import write_playlist

WANT_CONTENT = """<?xml version="1.0"?>
<AUDIOFILE_PLAYLIST>
   <PLAYLIST1>
      <Entry Folder="Audio Tracks"/>
      <Entry Code001="track1.wav"/>
      <Entry Code002="track2.wav"/>
      <Entry Code003="track3.wav"/>
      <Entry Code004="track4.wav"/>
      <Entry Code005="track5.wav"/>
      <Entry Code006="track6.wav"/>
      <Entry Code007="track7.wav"/>
      <Entry Code008="track8.wav"/>
      <Entry Code009="track9.wav"/>
      <Entry Code010="track10.wav"/>
      <Entry Code011="track11.wav"/>
      <Entry Code012="track12.wav"/>
      <Entry Code013="track13.wav"/>
      <Entry Code014="track14.wav"/>
      <Entry Code015="track15.wav"/>
      <Entry Code016="track16.wav"/>
      <Entry Code017="track17.wav"/>
      <Entry Code018="track18.wav"/>
      <Entry Code019="track19.wav"/>
      <Entry Code020="track20.wav"/>
      <Entry Code021="track21.wav"/>
      <Entry Code022="track22.wav"/>
      <Entry Code023="track23.wav"/>
      <Entry Code024="track24.wav"/>
      <Entry Code025="track25.wav"/>
      <Entry Code026="track26.wav"/>
      <Entry Code027="track27.wav"/>
      <Entry Code028="track28.wav"/>
      <Entry Code029="track29.wav"/>
      <Entry Code030="track30.wav"/>
      <Entry Code031="track31.wav"/>
      <Entry Code032="track32.wav"/>
      <Entry Code033="track33.wav"/>
      <Entry Code034="track34.wav"/>
      <Entry Code035="track35.wav"/>
      <Entry Code036="track36.wav"/>
      <Entry Code037="track37.wav"/>
      <Entry Code038="track38.wav"/>
      <Entry Code039="track39.wav"/>
      <Entry Code040="track40.wav"/>
      <Entry Code041="track41.wav"/>
      <Entry Code042="track42.wav"/>
      <Entry Code043="track43.wav"/>
      <Entry Code044="track44.wav"/>
      <Entry Code045="track45.wav"/>
      <Entry Code046="track46.wav"/>
      <Entry Code047="track47.wav"/>
      <Entry Code048="track48.wav"/>
      <Entry Code049="track49.wav"/>
      <Entry Code050="track50.wav"/>
      <Entry Code051="track51.wav"/>
      <Entry Code052="track52.wav"/>
      <Entry Code053="track53.wav"/>
      <Entry Code054="track54.wav"/>
      <Entry Code055="track55.wav"/>
      <Entry Code056="track56.wav"/>
      <Entry Code057="track57.wav"/>
      <Entry Code058="track58.wav"/>
      <Entry Code059="track59.wav"/>
      <Entry Code060="track60.wav"/>
      <Entry Code061="track61.wav"/>
      <Entry Code062="track62.wav"/>
      <Entry Code063="track63.wav"/>
      <Entry Code064="track64.wav"/>
      <Entry Code065="track65.wav"/>
      <Entry Code066="track66.wav"/>
      <Entry Code067="track67.wav"/>
      <Entry Code068="track68.wav"/>
      <Entry Code069="track69.wav"/>
      <Entry Code070="track70.wav"/>
      <Entry Code071="track71.wav"/>
      <Entry Code072="track72.wav"/>
      <Entry Code073="track73.wav"/>
      <Entry Code074="track74.wav"/>
      <Entry Code075="track75.wav"/>
      <Entry Code076="track76.wav"/>
      <Entry Code077="track77.wav"/>
      <Entry Code078="track78.wav"/>
      <Entry Code079="track79.wav"/>
      <Entry Code080="track80.wav"/>
      <Entry Code081="track81.wav"/>
      <Entry Code082="track82.wav"/>
      <Entry Code083="track83.wav"/>
      <Entry Code084="track84.wav"/>
      <Entry Code085="track85.wav"/>
      <Entry Code086="track86.wav"/>
      <Entry Code087="track87.wav"/>
      <Entry Code088="track88.wav"/>
      <Entry Code089="track89.wav"/>
      <Entry Code090="track90.wav"/>
      <Entry Code091="track91.wav"/>
      <Entry Code092="track92.wav"/>
      <Entry Code093="track93.wav"/>
      <Entry Code094="track94.wav"/>
      <Entry Code095="track95.wav"/>
      <Entry Code096="track96.wav"/>
      <Entry Code097="track97.wav"/>
      <Entry Code098="track98.wav"/>
      <Entry Code099="track99.wav"/>
      <Entry Code100="track100.wav"/>
      <Entry Code101="track101.wav"/>
      <Entry Code102="track102.wav"/>
      <Entry Code103="track103.wav"/>
      <Entry Code104="track104.wav"/>
      <Entry Code105="track105.wav"/>
      <Entry Code106="track106.wav"/>
      <Entry Code107="track107.wav"/>
      <Entry Code108="track108.wav"/>
      <Entry Code109="track109.wav"/>
      <Entry Code110="track110.wav"/>
      <Entry Code111="track111.wav"/>
      <Entry Code112="track112.wav"/>
      <Entry Code113="track113.wav"/>
      <Entry Code114="track114.wav"/>
      <Entry Code115="track115.wav"/>
      <Entry Code116="track116.wav"/>
      <Entry Code117="track117.wav"/>
      <Entry Code118="track118.wav"/>
      <Entry Code119="track119.wav"/>
      <Entry Code120="track120.wav"/>
      <Entry Code121="track121.wav"/>
      <Entry Code122="track122.wav"/>
      <Entry Code123="track123.wav"/>
      <Entry Code124="track124.wav"/>
      <Entry Code125="track125.wav"/>
      <Entry Code126="track126.wav"/>
      <Entry Code127="track127.wav"/>
   </PLAYLIST1>
   <SYSTEM>
      <Entry UseDigitalInputD0="FALSE"/>
      <Entry StopCode="19"/>
      <Entry SwapChannels="FALSE"/>
      <Entry SDRAMTest="FALSE"/>
   </SYSTEM>
</AUDIOFILE_PLAYLIST>
"""


def test_playlist(tmpdir):
    """Test writing an XML playlist for AudioFile."""
    # sanity check our desired data
    tree = ET.fromstring(WANT_CONTENT)
    assert tree.tag == 'AUDIOFILE_PLAYLIST'
    assert len(tree) == 2
    assert tree[0].tag == 'PLAYLIST1'
    assert tree[1].tag == 'SYSTEM'

    out_fname = op.join(str(tmpdir), 'Playlist.xml')

    # bad output filename
    with pytest.raises(ValueError, match='playlist_fname must end with'):
        write_playlist(['1.wav'], out_fname[:-1])
    with pytest.raises(TypeError, match='expected str'):
        write_playlist(['1.wav'], 1.)

    # bad input filenames
    with pytest.raises(TypeError, match='fnames must be a list of str'):
        write_playlist('foo')
    with pytest.raises(TypeError, match='Entries in fnames must be str'):
        write_playlist([1.])
    with pytest.raises(ValueError, match='in fnames cannot have slashes'):
        write_playlist(['/foo.wav'])
    with pytest.raises(ValueError, match='must end with .wav'):
        write_playlist(['foo.wa'])

    # bad subdirectory
    for bad in (1., 'me/you'):
        with pytest.raises(ValueError, match='subdirectory must'):
            write_playlist(['foo.wav'], out_fname, subdirectory=bad)

    # actually write a list
    fnames = ['track%d.wav' % (ii + 1,) for ii in range(127)]
    write_playlist(fnames, out_fname)
    assert op.isfile(out_fname)
    with open(out_fname, 'r') as fid:
        data = fid.read()
    assert data == WANT_CONTENT

    # no overwrite
    with pytest.raises(OSError, match='File Playlist.xml exists'):
        write_playlist(fnames, out_fname)
