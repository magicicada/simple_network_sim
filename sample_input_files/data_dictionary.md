This file will contain brief descriptions of the sample file formats in the sample_input_files directory, or other descriptions where suitable.  This will likely be deprecated by a later move to HDF5 I/O, but for development for now we need to use the data in the formats we have it (as of Thursday, May 7, 2020)

# Files

## paramsFile

This contains epidemiological rate parameters for movement within the compartmental model - not all rates are included, since some can be inferred from the ones that are provided.
compartment_escape entries are the rate at which individual `s` leave that compartment *to any other* per timestep - e.g. `e_escape:0.427` says that the rate at which individuals leave the `E` state is 0.427
Explicit compartment transfer rates rates from the first compartment in the name to the second - e.g: `i_to_h:0.15` says the rate of movement from `I` to `H` is 0.15
*TODO* for the future - amend the format of this file to make chaining compartmental structure even easier - that is, e.g., `i_to_h:0.15` might become `(I,H):0.15` - not doing for now because of likely change to HDF5

## paramsAgeStructured
This contains epidemiological rates as for [paramsFile](#paramsFile), but for young (lines starting with `y`), mature (lines starting with `m`), and old (lines starting `o`) individuals.

## sample_hb2019_pop_est_2018.sampleCSV
This file contains age-and-sex-stratified population numbers by geographic unit (here Scottish health board areas, but same format is available for other geographic units).  Columns are:
Health_Board,Sex,Total_across_age,Young,Medium,Old

* Health_board is the name of the geographic unit.
* Sex is male or female.
* Total_across_age is the total population of the sex in this row across all ages
* Young is the population of the sex in this row of people 16 and under
* Mature is the population of the sex in this row from ages 17-69
* Old is the population of the sex in this row from ages 70 upwards


## sample_scotHB_commute_moves_wu01.sampleCSV
This contains origin-destination flow data during peacetime for health boards in Scotland, derived from the dataset [wu01uk](https://www.nomisweb.co.uk/census/2011/wu01uk).
This is essentially a weighted edge list.
* First column is the source location label
* Second column is the destination location label
* Third column is the number of individuals undertaking that journey as reported in wu01uk

## sample_20200327_comix_social_contacts.sampleCSV
This is a sample square matrix of mixing - each column and row header is an age category.
