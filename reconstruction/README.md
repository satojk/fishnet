A previous version of data/scraper.py used .strip('/black') to remove the 
suffix '/black' from game_ids. This was the result of a misinterpretation of 
how strip works. The new version now uses .replace('/black', '').

The script in this page attempted to reconstruct game_ids that were missing 
characters. logs.txt is the logs file from the full running of the script.
