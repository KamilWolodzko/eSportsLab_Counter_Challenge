# eSportsLab_Counter_Challenge
# From esportsLABgg Counter-Strike Data Challenge technical description of the task

Task Details
Your main task is to write a script, which takes as input a CSV file with CS:GO matches grenade throws data. 
The provided datafiles contain features of selected grenades thrown by either team – terrorists (T) or
counter-terrorists (CT) during matches on two major competitive maps (de inferno and de mirage).
Your task is to create a python binary classifier for labeling grenade throws as correct/incorrect.
Your solution should be a single python script which reads from the command-line input the name of a
file with grenade features (can use e.g. ’argparse’ library). Your script name should be ’classify.ph’, if the
test grenades filename is ’test.csv’, we will execute your script as follows
python classify.py test.csv
The role of your script is to perform classification of each of the grenade (described using a single row of
features) in the CSV input file, and append the classification result (boolean value TRUE = correct
throw/FALSE = incorrect throw) to the provided input file as an additional column ’RESULT’ and
modify it in-place.

Problem Statement
GRENADE FEATURES Each grenade throw recorded in the input CSV file (single row) is described using
the following set of features (the CSV contains also set of id values for our internal use: demo id, demo round id,
round start tick, weapon fire id)
• team: T – terrorists, CT – counter-terrorists;
• (detonation raw x, detonation raw y , detonation raw z ): grenade detonation raw coordinates;
• (throw from raw x, throw from raw y, throw from raw z ): raw coordinates of the player when the grenade is
being thrown;
• throw tick: the exact tick (unit of game time, 128 ticks per second, counted from the beginning of the game),
when the grenade is being thrown;
• detonation tick: the exact tick, when the grenade is being detonated;
• TYPE: type of the grenade (smoke, flashbang, molotov);
• map name: map on which the match was played (de inferno, de mirage);
LABELED TRAINING DATA To design a successful classifier you may use training datasets. We provide
two training datasets consisting of grenade throw data from two major competitive maps: de inferno and de mirage,
each grenade throw is labeled using a boolean value. The meaning of the values in column ’LABEL’ is either correct
throw (TRUE) or incorrect throw (FALSE). We provide two sets of labeled training data:
• train-grenades-de inferno.csv: features of 354 grenade throws on de inferno map;
• train-grenades-de mirage.csv: features of 370 grenade throws on de mirage map;

