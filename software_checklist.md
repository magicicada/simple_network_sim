# SCRC Software checklist

This checklist is part of ongoing work on a model scoresheet for SCRC models. It relates to software implementation, and assumes that other documents cover questions about model validation, data provenance and quality, quality of science, and policy readiness.

## Software Details

### Model / software name

> simple_network_sim

### Date

> 20200727

### Version identifier

> Version not yet specified

## Overall statement

Do we have sufficient confidence in the correctness of the software to trust the results? Yes / Yes with caveats / No

This is your overall judgement on the level of confidence based on all the aspects of the checklist. There is no formulaic way to arrive at this overall assessment based on the individual checklist answers but please explain how the measures in place combine to reach this level of confidence and make clear any caveats (eg applies for certain ways of using the software and not others).

> The code is well crafted and organised, with automated unit and regression testing via continuous integration. Execution environment is specified using [conda](https://docs.conda.io/en/latest/). It is planned that data integrity and provenance for reproducible model runs will be assured by use of the SCRC data pipeline, which is not yet fully available. Further work will be required to integrate with this. There are some areas where important improvements can are still being made (see below and the [project board](https://github.com/orgs/ScottishCovidResponse/projects/6).

## Checklist

Please use a statement from this list: "Sufficiently addressed", "Some work remaining or caveats", or "Needs to be addressed" to begin each response.

Additionally, for each question please explain the situation and include any relevant links (eg tool dashboards, documentation). The sub bullet points are to make the scope of the question clear and should be covered if relevant but do not have to be answered individually.

### Can a run be repeated and reproduce exactly the same results?

- How is stochasticity handled?
- Is sufficient meta-data logged to enable a run to be reproduced: Is the exact code version recorded (and whether the repository was "clean"), including versions of dependent libraries (e.g. an environment.yml file or similar) along with all command line arguments and the content of any configuration files?
- Is there up-to-date documentation which explains precisely how to run the code to reproduce existing results?

> - [x] Sufficiently addressed
> - [ ] Some work remaining or caveats
> - [ ] Needs to be addressed
>
> Stochasticity is handled by using a seed from the data pipeline to generate a “random state” object which is passed to functions that generate random numbers. Git url and sha are saved when the software is run. Files that specify the conda environment are present: [spec-file.txt](spec-file.txt) and [environment.yml](environment.yml). Documentation explaining how to run the software with some test data can be found in the [readme](readme.md). We have not yet produced "real" results based on provenanced, reliable source data - this work is ongoing and reproducibility will be assured by a combination of this software and the SCRC data pipeline.

### Are there appropriate tests?  (And are they automated?)

- Are there unit tests? What is covered?
- System and integration tests?  Automated model validation tests?
- Regression tests? (Which show whether changes to the code lead to changes in the output. Changes to the model will be expected to change the output, but many other changes, such as refactoring and adding new features, should not. Having these tests gives confidence that the code hasn't developed bugs due to unintentional changes.)
- Is there CI?
- Is everything you need to run the tests (including documentation) in the repository (or the data pipeline where appropriate)?

> - [x] Sufficiently addressed
> - [ ] Some work remaining or caveats
> - [ ] Needs to be addressed
>
> [Unit tests and regression tests](README.md#Tests) are present ([![Code Coverage](https://codecov.io/github/ScottishCovidResponse/simple_network_sim/coverage.svg?branch=master&token=)](https://codecov.io/gh/ScottishCovidResponse/simple_network_sim)), run via continuous integration [![Build Status](https://travis-ci.org/ScottishCovidResponse/simple_network_sim.svg?branch=master)](https://travis-ci.org/ScottishCovidResponse/simple_network_sim). Automated model validation tests are not, but could be added. Jupyter notebooks in the repository are not currently tested - this is difficult at present as it relies on data we cannot share and is computationally expensive - ultimately the notebooks should at least be run end-to-end as part of testing.

### Are the scientific results of runs robust to different ways of running the code?

- Running on a different machine?
- With different number of processes?
- With different compilers and optimisation levels?
- Running in debug mode?

(We don't require bitwise identical results here, but the broad conclusions after looking at the results of the test case should be the same.) 

> - [x] Sufficiently addressed
> - [ ] Some work remaining or caveats
> - [ ] Needs to be addressed
>
> This applies to our test data and (can’t guarantee that it will run on any system) but regression tests have been run (informally) on a range of platforms (e.g. HPC, Windows 10, OSX). We do not use multiple processes at present - this will change very soon.

### Has any sort of automated code checking been applied?

- For C++, this might just be the compiler output when run with "all warnings". It could also be more extensive static analysis. For other languages, it could be e.g. pylint, StaticLint.jl, etc.
- If there are possible issues reported by such a tool, have they all been either fixed or understood to not be important?

> - [x] Sufficiently addressed
> - [ ] Some work remaining or caveats
> - [ ] Needs to be addressed
>
> We use [pylint, automated by codacy](README.md#static-analysis), to check code going into the repository. We make some exceptions to the pylint defaults – these are [specified in the repo](.pylintrc). New and altered code is checked and issues dealt with prior to being integrated into the `master` branch.

### Is the code clean, generally understandable and readable and written according to good software engineering principles?

- Is it modular?  Are the internal implementation details of one module hidden from other modules?
- Commented where necessary?
- Avoiding red flags such as very long functions, global variables, copy and pasted code, etc.?

> - [x] Sufficiently addressed
> - [ ] Some work remaining or caveats
> - [ ] Needs to be addressed
>
> Code is modular, documented, low in duplication and well laid out.

### Is there sufficient documentation?

- Is there a readme?
- Does the code have user documentation?
- Does the code have developer documentation?
- Does the code have algorithm documentation? e.g. something that describes how the model is actually simulated, or inference is performed?
- Is all the documentation up to date?

> - [ ] Sufficiently addressed
> - [x] Some work remaining or caveats
> - [ ] Needs to be addressed
>
> We have (slightly outdated) [user](README.md), [developer](https://readthedocs.org/projects/simple-network-sim/badge/?version=latest) and (slightly outdated) [algorithm](model_overview_simple_network_sim.md) documentation. A GitHub issue has been raised to address the areas of documentation that need to be updated.

### Is there suitable collaboration infrastructure?

- Is the code in a version-controlled repository?
- Is there a license?
- Is an issue tracker used?
- Are there contribution guidelines?

> - [x] Sufficiently addressed
> - [ ] Some work remaining or caveats
> - [ ] Needs to be addressed
>
> We use a [GitHub repository](https://github.com/ScottishCovidResponse/simple_network_sim), with a [BSD license](LICENSE.txt), issue tracking and [contribution guidelines](contributing.md).

### Are software dependencies listed and of appropriate quality?

> - [ ] Sufficiently addressed
> - [ ] Some work remaining or caveats
> - [x] Needs to be addressed
>
> Dependencies are specified in a conda [spec-file](spec-file.txt). We are investigating how best to handle `git` as a binary dependency in the conda paradigm, but this is not a major issue. We depend on the SCRC Data Pipeline and [SCRC Data Pipeline API](https://github.com/ScottishCovidResponse/data_pipeline_api) which are not yet sufficiently developed to ensure robust, reproducible operation - they are under intensive development and we expect to have this soon.

### Is input and output data handled carefully?

- Does the code use the data pipeline for all inputs and outputs?
- Is the code appropriately parameterized (i.e. have hard coded parameters been removed)?

> - [ ] Sufficiently addressed
> - [ ] Some work remaining or caveats
> - [x] Needs to be addressed
> 
> The SCRC Data Pipeline is not being used fully as it is under intensive development. We expect to have this soon. Code is parameterised.