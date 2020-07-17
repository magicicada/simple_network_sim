This file will contain brief descriptions of the sample file formats in the
sample_input_files directory, or other descriptions where suitable.  This will
likely be deprecated by a later move to HDF5 I/O, but for development for now
we need to use the data in the formats we have it (as of Thursday, May 7, 2020)

# Files

## human/compartment-transition/1/data.csv

This contains epidemiological rate parameters for movement within the
compartmental model split by ages. Each row represents a transition between two
compartmens within a specific age.

## human/compartment-transition/2/data.csv

Same as above, but this splits the asymptomatics into A (pre-symptomatic) and
A_2 (never symptomatic).

Use this with _human/infectious-compartments/2/data.csv_

## human/population/1/data.csv

This file has the population in each region (Scottish healthboards) stratified
by age and sex. Each row has the following columns:

* Health_board is the name of the geographic unit.
* Sex is male or female.
* Age is the age group (eg.: \[0,17) is the age group from 0 to 17 years old)
* Total is the number of people inside the the categories described by the other columns

## human/commutes/1/data.csv

This contains origin-destination flow data during peacetime for health boards
in Scotland, derived from the dataset
[wu01uk](https://www.nomisweb.co.uk/census/2011/wu01uk).

This is essentially a weighted edge list.
* First column is the source location label
* Second column is the destination location label
* Third column is the number of individuals undertaking that journey as
  reported in wu01uk
* Fourth column is a multiplier that gives control over how much dampening
  factors impact this edge

## human/full-mixing-matrix/1/data.csv

This comes from is from
https://cmmid.github.io/topics/covid19/comix-impact-of-physical-distance-measures-on-transmission-in-the-UK.html,
and includes all imputed contacts (both face-to-face conversation and physical
contact). This is the entire comix matrix, for all age groups, which is not
compatible with the population data we have this far. Look the
_human/mixing-matrix/1/data.csv_ below.

## human/mixing-matrix/1/data.csv

This is a sample *simplified* square matrix describing age mixing between
aggregated age categories, generated from _human/full-mixing-matrix/1/data.csv_
by summing over the columns that need to be aggregated, and then summing
(weighted by the approximate population in each category) over the rows that
need to be aggregated.  This includes both face-to-face and physical contacts,
and is not yet down-weighted to approximate infection probability.  Age
proportion estimated from
https://www.ons.gov.uk/peoplepopulationandcommunity/populationandmigration/populationestimates/articles/overviewoftheukpopulation/august2019.

## human/movement-multpliers/1/data.csv

This table has three columns: Time, Movement Multiplier and Contact Multiplier.
For instance, if the file looks like this:

```
Time,Movement_Multiplier,Contact_Multiplier
0,1.0,1.0
3,0.7,0.4
10,0.9,0.6
```

The movement multiplier controls how much the movements between nodes is scaled
up or down, whereas the contact multiplier controls how to the within node
contact is increased or slowed down over time.


## human/infection-probability/1/data.csv

The values in this table are the probability that a contact between an
infectious person with a susceptible person will result in a transmission over
time.

## human/infectious-compartments/1/data.csv

A list of compartments that are considered infectious.

## human/infectious-compartments/2/data.csv

Same as above, but includes the A_2 compartment

## human/initial-infections/1/data.csv

A list with seed nodes, where the outbreak will start for a given simulation.
This table contains both the region ID and number of infected people.

## human/stochastic-mode/1/data.csv

Whether to use the stochastic or deterministic version of the model. A
production run will usually be stochastic, but the deterministic run can be
useful during development.

## human/random-seed/1/data.csv

The seed to be used by the model during the stochastic run. This is ignored for
deterministic runs.
