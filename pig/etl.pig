-- ***************************************************************************
-- TASK
-- Aggregate events into features of patient and generate training, testing data for mortality prediction.
-- Steps have been provided to guide you.
-- You can include as many intermediate steps as required to complete the calculations.
-- ***************************************************************************

-- ***************************************************************************
-- TESTS
-- To test, please change the LOAD path for events and mortality to ../../test/events.csv and ../../test/mortality.csv
-- 6 tests have been provided to test all the subparts in this exercise.
-- Manually compare the output of each test against the csv's in test/expected folder.
-- ***************************************************************************

-- register a python UDF for converting data into SVMLight format
REGISTER utils.py USING jython AS utils;

-- load events file
--events = LOAD '../../data/events.csv' USING PigStorage(',') AS (patientid:int, eventid:chararray, eventdesc:chararray, timestamp:chararray, value:float);
events = LOAD '../../code/sample_test/sample_events.csv' USING PigStorage(',') AS (patientid:int, eventid:chararray, eventdesc:chararray, timestamp:chararray, value:float);

-- select required columns from events
events = FOREACH events GENERATE patientid, eventid, ToDate(timestamp, 'yyyy-MM-dd') AS etimestamp, value;

-- load mortality file
--mortality = LOAD '../../data/mortality.csv' USING PigStorage(',') as (patientid:int, timestamp:chararray, label:int);
mortality = LOAD '../../code/sample_test/sample_mortality.csv' USING PigStorage(',') as (patientid:int, timestamp:chararray, label:int);

mortality = FOREACH mortality GENERATE patientid, ToDate(timestamp, 'yyyy-MM-dd') AS mtimestamp, label;

--To display the relation, use the dump command e.g. DUMP mortality;

-- ***************************************************************************
-- Compute the index dates for dead and alive patients
-- ***************************************************************************
eventswithmort = JOIN events by patientid LEFT OUTER, mortality by patientid; -- perform join of events and mortality by patientid;
eventswithmort2 = FOREACH eventswithmort GENERATE events::patientid AS patientid, events::eventid AS eventid, events::etimestamp AS etimestamp, events::value AS value, mortality::mtimestamp AS mtimestamp, mortality::label AS label;

-- detect the events of dead patients and create it of the form (patientid, eventid, value, label, time_difference) where time_difference is the days between index date and each event timestamp
dead_events = FILTER eventswithmort2 BY (label is not null);
dead_events2 = FOREACH dead_events GENERATE patientid AS patientid, eventid AS eventid, etimestamp AS etimestamp, value AS value, mtimestamp AS mtimestamp, label AS label;
dead_indexes = FOREACH mortality GENERATE patientid AS patientid, SubtractDuration(mtimestamp,'P30D') AS index_date;
dead_join = JOIN dead_events2 BY patientid, dead_indexes BY patientid;
deadevents = FOREACH dead_join GENERATE dead_events2::patientid AS patientid, dead_events2::eventid AS eventid, dead_events2::value AS value, dead_events2::label AS label, DaysBetween(dead_indexes::index_date, dead_events2::etimestamp) AS time_difference;

-- detect the events of alive patients and create it of the form (patientid, eventid, value, label, time_difference) where time_difference is the days between index date and each event timestamp
alive_filter = FILTER eventswithmort2 BY (label is null);
alive_filter_rename = FOREACH alive_filter GENERATE patientid AS patientid, eventid AS eventid, etimestamp AS etimestamp, value AS value, 0 AS label;
alivegroup = GROUP alive_filter_rename BY (patientid, eventid, etimestamp);
alive_indexes = FOREACH alivegroup GENERATE group.patientid, MAX(alive_filter_rename.etimestamp) AS index_date;
alive_join = JOIN alive_filter_rename BY patientid, alive_indexes BY patientid;
aliveevents = FOREACH alive_join GENERATE alive_filter_rename::patientid AS patientid, alive_filter_rename::eventid AS eventid, alive_filter_rename::value AS value, alive_filter_rename::label AS label, DaysBetween(alive_indexes::index_date, alive_filter_rename::etimestamp) AS time_difference;

--TEST-1
deadevents = ORDER deadevents BY patientid, eventid;
aliveevents = ORDER aliveevents BY patientid, eventid;
STORE aliveevents INTO 'aliveevents' USING PigStorage(',');
STORE deadevents INTO 'deadevents' USING PigStorage(',');



-- ***************************************************************************
-- Filter events within the observation window and remove events with missing values
-- ***************************************************************************
eventsall = UNION deadevents, aliveevents;
filtered = FILTER eventsall BY (value is not null) AND (time_difference <= 2000) AND (time_difference >= 0);
-- contains only events for all patients within the observation window of 2000 days and is of the form (patientid, eventid, value, label, time_difference)

--TEST-2
filteredgrpd = GROUP filtered BY 1;
filtered = FOREACH filteredgrpd GENERATE FLATTEN(filtered);
filtered = ORDER filtered BY patientid, eventid,time_difference;
STORE filtered INTO 'filtered' USING PigStorage(',');

-- ***************************************************************************
-- Aggregate events to create features
-- ***************************************************************************
featureswithid = GROUP filtered BY (patientid, eventid);
featureswithid2 = FOREACH featureswithid GENERATE group.patientid AS patientid, group.eventid AS eventid, COUNT(filtered.value) AS featurevalue;
-- for group of (patientid, eventid), count the number of  events occurred for the patient and create relation of the form (patientid, eventid, featurevalue)

--TEST-3
featureswithid = ORDER featureswithid BY patientid, eventid;
STORE featureswithid INTO 'features_aggregate' USING PigStorage(',');

-- ***************************************************************************
-- Generate feature mapping
-- ***************************************************************************
feature_names = FOREACH filtered GENERATE eventid;
feature_names_distinct = DISTINCT feature_names;
--DESCRIBE feature_names;
all_features_dist = RANK feature_names_distinct;
all_features = FOREACH all_features_dist GENERATE $0 AS idx, $1 AS eventid;
-- compute the set of distinct eventids obtained from previous step, sort them by eventid and then rank these features by eventid to create (idx, eventid). Rank should start from 0.

-- store the features as an output file
STORE all_features INTO 'features' using PigStorage(' ');

features_pre = JOIN featureswithid2 BY eventid, all_features BY eventid;
features = foreach features_pre GENERATE featureswithid2::patientid AS patientid, all_features::idx AS idx, featureswithid2::featurevalue AS featurevalue;
-- perform join of featureswithid and all_features by eventid and replace eventid with idx. It is of the form (patientid, idx, featurevalue)

--TEST-4
features = ORDER features BY patientid, idx;
STORE features INTO 'features_map' USING PigStorage(',');

-- ***************************************************************************
-- Normalize the values using min-max normalization
-- Use DOUBLE precision
-- ***************************************************************************
maxvalues_grp = GROUP features BY (idx);
maxvalues = foreach maxvalues_grp GENERATE idx AS idx, MAX(features.featurevalue) AS maxvalue;
-- group events by idx and compute the maximum feature value in each group. I t is of the form (idx, maxvalue)

normalized = JOIN features BY idx, maxvalues BY idx;
-- join features and maxvalues by idx

features = FOREACH normalized GENERATE features.patientid, features.idx, (1.0*(features.featurevalue) / (maxvalues.maxvalue)) AS normalizedfeaturevalue;
-- compute the final set of normalized features of the form (patientid, idx, normalizedfeaturevalue)

--TEST-5
features = ORDER features BY patientid, idx;
STORE features INTO 'features_normalized' USING PigStorage(',');

-- ***************************************************************************
-- Generate features in svmlight format
-- features is of the form (patientid, idx, normalizedfeaturevalue) and is the output of the previous step
-- e.g.  1,1,1.0
--  	 1,3,0.8
--	     2,1,0.5
--       3,3,1.0
-- ***************************************************************************

grpd = GROUP features BY patientid;
grpd_order = ORDER grpd BY $0;
features = FOREACH grpd_order
{
    sorted = ORDER features BY idx;
    generate group as patientid, utils.bag_to_svmlight(sorted) as sparsefeature;
}

-- ***************************************************************************
-- Split into train and test set
-- labels is of the form (patientid, label) and contains all patientids followed by label of 1 for dead and 0 for alive
-- e.g. 1,1
--	2,0
--      3,1
-- ***************************************************************************

labels = FOREACH filtered GENERATE patientid, label;
labels = distinct labels;
-- create it of the form (patientid, label) for dead and alive patients

--Generate sparsefeature vector relation
samples = JOIN features BY patientid, labels BY patientid;
samples = DISTINCT samples PARALLEL 1;
samples = ORDER samples BY $0;
samples = FOREACH samples GENERATE $3 AS label, $1 AS sparsefeature;

--TEST-6
STORE samples INTO 'samples' USING PigStorage(' ');

-- randomly split data for training and testing
DEFINE rand_gen RANDOM('6505');
samples = FOREACH samples GENERATE rand_gen() as assignmentkey, *;
SPLIT samples INTO testing IF assignmentkey <= 0.20, training OTHERWISE;
training = FOREACH training GENERATE $1..;
testing = FOREACH testing GENERATE $1..;

-- save training and tesing data
STORE testing INTO 'testing' USING PigStorage(' ');
STORE training INTO 'training' USING PigStorage(' ');
